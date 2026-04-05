import os
# 1. Memory Optimization (Prevents "Sudden Disappearance" crashes)
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507" 
DATASET_FILE = "all_data_new_reason5.jsonl" 
OUTPUT_DIR = "decompiler-distil-best"  # Folder for best model

# Hyperparameters
MAX_SEQ_LENGTH = 90112 
CHUNK_SIZE = 1024      # Lowered to 1024 for stability on B200
ALPHA = 0.5   
TEMP = 2.0    
LORA_R = 32   
LORA_ALPHA = 32

# ------------------------------------------------------------------------
# 3. Optimized Trainer (Chunked Loss + Logging)
# ------------------------------------------------------------------------
class ChunkedDualStreamTrainer(Trainer):
    # ------------------------------------------------------------------
    # 1. Training Step (Optimized for B200 Memory)
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # POP keys to separate them from standard model inputs
        # (We use .pop() so they don't get passed to the backbone later)
        teacher_probs = inputs.pop("teacher_probs")
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # A. Backbone Extraction
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            causal_lm = model.base_model.model 
        else:
            causal_lm = model

        if hasattr(causal_lm, "model"):
            backbone = causal_lm.model 
            lm_head = causal_lm.lm_head
        elif hasattr(causal_lm, "transformer"):
            backbone = causal_lm.transformer
            lm_head = causal_lm.lm_head
        else:
            backbone = causal_lm.model
            lm_head = causal_lm.lm_head

        # B. Forward Pass (Backbone Only - Low Memory)
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        hidden_states = outputs.last_hidden_state 
        
        # C. Prepare Data
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_teacher = teacher_probs[..., 1:].contiguous()
        
        seq_len = shift_labels.size(1)
        device = input_ids.device
        
        # Initialize as tensors for graph connectivity
        total_sft_loss = torch.tensor(0.0, device=device)
        total_kd_loss = torch.tensor(0.0, device=device)
        sft_count = 0
        kd_count = 0
        
        # D. Chunked Projection Loop (1024 tokens at a time)
        for i in range(0, seq_len, CHUNK_SIZE):
            j = min(i + CHUNK_SIZE, seq_len)
            
            # Slice
            h_chunk = shift_hidden[:, i:j, :]
            lbl_chunk = shift_labels[:, i:j]
            tp_chunk = shift_teacher[:, i:j]
            
            # Project to Logits
            logits_chunk = lm_head(h_chunk)
            
            flat_logits = logits_chunk.view(-1, logits_chunk.size(-1))
            flat_labels = lbl_chunk.reshape(-1)
            flat_teacher = tp_chunk.reshape(-1)

            # SFT Loss
            valid_mask = (flat_labels != -100)
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='sum')
                total_sft_loss = total_sft_loss + loss
                sft_count += valid_mask.sum().item()
            
            # KD Loss
            kd_mask = (flat_teacher > 1e-6) & valid_mask
            if kd_mask.sum() > 0:
                log_probs = F.log_softmax(flat_logits / TEMP, dim=-1)
                gather_idx = flat_labels.clamp(min=0)
                selected = torch.gather(log_probs, -1, gather_idx.unsqueeze(-1)).squeeze(-1)
                
                kd_loss_chunk = -(flat_teacher * selected * kd_mask).sum()
                total_kd_loss = total_kd_loss + kd_loss_chunk
                kd_count += kd_mask.sum().item()
            
            del logits_chunk, h_chunk, flat_logits, flat_labels, flat_teacher
        
        # E. Normalize & Connect Graph
        if sft_count > 0:
            sft_loss = total_sft_loss / sft_count
        else:
            sft_loss = (hidden_states * 0).sum()
            
        if kd_count > 0:
            kd_loss = total_kd_loss / kd_count
        else:
            kd_loss = (hidden_states * 0).sum()
        
        total_loss = (1.0 - ALPHA) * sft_loss + ALPHA * kd_loss
        
        # F. Logging (Only log during training, skip during eval to avoid spam)
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            def val(x): return x.item() if torch.is_tensor(x) else x
            self.log({
                "loss/sft": val(sft_loss),
                "loss/kd": val(kd_loss),
                "tokens/sft": sft_count,
                "tokens/kd": kd_count,
            })

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # 2. Evaluation Step (Prevents OOM during Eval)
    # ------------------------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom Evaluation Loop.
        1. Prepares inputs (moves to GPU).
        2. Clones inputs because compute_loss pops keys.
        3. Uses Chunked Loss to prevent OOM.
        """
        # Ensure inputs are on the correct device
        inputs = self._prepare_inputs(inputs)
        
        # Shallow copy dictionary + Clone tensors to prevent side-effects 
        # (Since compute_loss uses .pop(), it destroys the input dict for future calls)
        inputs_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs_copy, return_outputs=False)
            
        # Return tuple required by Trainer: (loss, logits, labels)
        # We return None for logits/labels to save massive memory
        return (loss, None, None)

# ------------------------------------------------------------------------
# 4. Alignment & Data Processing
# ------------------------------------------------------------------------
def align_teacher_probs_to_student(teacher_logprobs, student_encodings, text_content):
    char_probs = [0.0] * len(text_content)
    curr = 0
    for t in teacher_logprobs:
        p = np.exp(t['logprob'])
        l = len(t['token'])
        end = min(curr + l, len(text_content))
        for k in range(curr, end): char_probs[k] = p
        curr += l
    aligned = []
    for s, e in student_encodings.offset_mapping:
        if s == e: aligned.append(0.0); continue
        span = char_probs[s:e]
        aligned.append(sum(span)/len(span) if span else 0.0)
    return aligned

def process_data_dual_stream(samples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "teacher_probs": [], "length": []}
    
    for i in range(len(samples['source'])):
        # 1. Prepare Prompt
        prompt = f"<|im_start|>system\nYou are a decompiler.<|im_end|>\n<|im_start|>user\n{samples['assembly'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Safety Check: If prompt alone is bigger than context, skip immediately
        if prompt_len >= MAX_SEQ_LENGTH - 10: 
            # -10 buffer to ensure at least SOME tokens are left for generation
            continue 

        # -------------------------------------------------------
        # STREAM A: DeepSeek (SFT Only)
        # -------------------------------------------------------
        ds_tokens = tokenizer(
            prompt + samples['reasoning'][i], 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            add_special_tokens=False
        )
        ds_labels = list(ds_tokens["input_ids"])
        
        # Mask the prompt
        # If prompt_len > len(ds_labels), the entire sequence is masked!
        mask_len = min(len(ds_labels), prompt_len)
        ds_labels[:mask_len] = [-100] * mask_len
        
        # CRITICAL FILTER: Check if there is anything left to predict
        # If all labels are -100, the answer was truncated off. Skip it.
        if all(l == -100 for l in ds_labels):
            continue

        model_inputs["input_ids"].append(ds_tokens["input_ids"])
        model_inputs["attention_mask"].append(ds_tokens["attention_mask"])
        model_inputs["labels"].append(ds_labels)
        model_inputs["teacher_probs"].append([0.0] * len(ds_tokens["input_ids"]))
        model_inputs["length"].append(len(ds_tokens["input_ids"]))

        # -------------------------------------------------------
        # STREAM B: Kimi (Distillation)
        # -------------------------------------------------------
        kimi_text = "".join([t['token'] for t in samples['logprobs'][i]])
        if kimi_text.strip():
            kimi_tokens = tokenizer(
                prompt + kimi_text, 
                truncation=True, 
                max_length=MAX_SEQ_LENGTH, 
                add_special_tokens=False, 
                return_offsets_mapping=True
            )
            k_labels = list(kimi_tokens["input_ids"])
            
            mask_len = min(len(k_labels), prompt_len)
            k_labels[:mask_len] = [-100] * mask_len
            
            # CRITICAL FILTER: Same check for Kimi stream
            if all(l == -100 for l in k_labels):
                continue
            
            # Alignment Logic
            resp_enc = tokenizer(kimi_text, return_offsets_mapping=True, add_special_tokens=False)
            aligned = align_teacher_probs_to_student(samples['logprobs'][i], resp_enc, kimi_text)
            
            final_probs = ([0.0]*prompt_len + aligned)[:len(kimi_tokens["input_ids"])]
            if len(final_probs) < len(kimi_tokens["input_ids"]): 
                final_probs += [0.0]*(len(kimi_tokens["input_ids"])-len(final_probs))
            
            model_inputs["input_ids"].append(kimi_tokens["input_ids"])
            model_inputs["attention_mask"].append(kimi_tokens["attention_mask"])
            model_inputs["labels"].append(k_labels)
            model_inputs["teacher_probs"].append([float(x) for x in final_probs])
            model_inputs["length"].append(len(kimi_tokens["input_ids"]))
            
    return model_inputs

def custom_collator(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
    
    teacher_probs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["teacher_probs"], dtype=torch.float32) for x in batch], 
        batch_first=True, 
        padding_value=0.0
    )

    rem = input_ids.size(1) % 64
    if rem != 0:
        pad = 64 - rem
        input_ids = F.pad(input_ids, (0, pad), value=0)
        attention_mask = F.pad(attention_mask, (0, pad), value=0)
        labels = F.pad(labels, (0, pad), value=-100)
        teacher_probs = F.pad(teacher_probs, (0, pad), value=0.0)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "teacher_probs": teacher_probs}

# ------------------------------------------------------------------------
# 5. Execution
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Dataset: {DATASET_FILE}")
    raw_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    
    # --- FIX 1: Create a tiny Validation Set (40 samples) ---
    # We need this to measure "Best Loss". 
    # We keep it small (1%) to avoid waiting hours for evaluation.
    dataset_splits = raw_dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = dataset_splits['train']
    eval_ds = dataset_splits['test']
    
    print("Processing Training Data...")
    train_formatted = train_ds.map(
        lambda x: process_data_dual_stream(x, tokenizer), 
        batched=True, 
        remove_columns=train_ds.column_names,
        load_from_cache_file=False 
    )
    
    print("Processing Validation Data...")
    eval_formatted = eval_ds.map(
        lambda x: process_data_dual_stream(x, tokenizer), 
        batched=True, 
        remove_columns=eval_ds.column_names,
        load_from_cache_file=False 
    )
    
    # Verify Columns
    if "teacher_probs" not in train_formatted.column_names:
        raise ValueError("Critical: 'teacher_probs' column missing.")

    print(f"Loading Model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto",
        attn_implementation="flash_attention_2" 
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    # Enable Input Gradients (Graph Safety)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --- FIX 2: Updated Arguments ---
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1, # Eval needs batch size too
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        
        # Strategies must match for Best Model Loading
        save_strategy="steps",
        eval_strategy="steps",        # <--- Added this
        save_steps=100,
        eval_steps=100,               # <--- Added this (evaluate every time we save)
        
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False, 
        report_to="none",
        dataloader_pin_memory=False
    )

    trainer = ChunkedDualStreamTrainer(
        model=model,
        args=args,
        train_dataset=train_formatted,
        eval_dataset=eval_formatted, # <--- Pass the validation set here
        data_collator=custom_collator,
    )

    print("Starting Training...")
    trainer.train()
    print(f"Saving Best Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        print("Pushing Model to HuggingFace Hub...")
        trainer.push_to_hub(commit_message="Initial commit", token=hf_token)