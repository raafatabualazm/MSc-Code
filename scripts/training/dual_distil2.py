import os
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
# Configuration
# ------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507" 
DATASET_FILE = "all_data_new_reason5.jsonl" 
OUTPUT_DIR = "decompiler-distil-best-2"

MAX_SEQ_LENGTH = 90112 
CHUNK_SIZE = 4096 
ALPHA = 0.3   
LORA_R = 32   
LORA_ALPHA = 32

# ------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------
class ChunkedDualStreamTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_probs = inputs.pop("teacher_probs")
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            causal_lm = model.base_model.model 
        else:
            causal_lm = model

        if hasattr(causal_lm, "model"):
            backbone = causal_lm.model 
            lm_head = causal_lm.lm_head
        else:
            backbone = causal_lm.transformer
            lm_head = causal_lm.lm_head

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state 
        
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_teacher = teacher_probs[..., 1:].contiguous()
        
        seq_len = shift_labels.size(1)
        device = input_ids.device
        
        total_sft_loss = torch.tensor(0.0, device=device)
        total_kd_loss = torch.tensor(0.0, device=device)
        sft_token_count = 0
        kd_weight_sum = torch.tensor(0.0, device=device) 
        
        for i in range(0, seq_len, CHUNK_SIZE):
            j = min(i + CHUNK_SIZE, seq_len)
            
            h_chunk = shift_hidden[:, i:j, :]
            lbl_chunk = shift_labels[:, i:j]
            tp_chunk = shift_teacher[:, i:j]
            
            logits_chunk = lm_head(h_chunk)
            
            flat_logits = logits_chunk.view(-1, logits_chunk.size(-1))
            flat_labels = lbl_chunk.reshape(-1)
            flat_teacher = tp_chunk.reshape(-1)

            valid_mask = (flat_labels != -100)
            
            if valid_mask.sum() > 0:
                sft_chunk_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='sum')
                total_sft_loss = total_sft_loss + sft_chunk_loss
                sft_token_count += valid_mask.sum().item()
            
            kd_mask = (flat_teacher > 1e-6) & valid_mask
            if kd_mask.sum() > 0:
                ce_per_token = F.cross_entropy(
                    flat_logits[kd_mask], 
                    flat_labels[kd_mask], 
                    reduction='none'
                )
                weights = flat_teacher[kd_mask]
                total_kd_loss = total_kd_loss + (weights * ce_per_token).sum()
                kd_weight_sum = kd_weight_sum + weights.sum()
            
            del logits_chunk, h_chunk, flat_logits, flat_labels, flat_teacher

        if sft_token_count > 0:
            sft_loss = total_sft_loss / sft_token_count
        else:
            sft_loss = (hidden_states * 0).sum() 

        if kd_weight_sum.item() > 0:
            kd_loss = total_kd_loss / kd_weight_sum
        else:
            kd_loss = (hidden_states * 0).sum()
        
        total_loss = (1.0 - ALPHA) * sft_loss + ALPHA * kd_loss
        
        if model.training and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss/sft": sft_loss.item(),
                "loss/kd": kd_loss.item(),
                "loss/total": total_loss.item(),
                "stats/sft_tokens": sft_token_count,
                "stats/kd_weight_sum": kd_weight_sum.item()
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        inputs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs_copy, return_outputs=False)
        return (loss, None, None)

# ------------------------------------------------------------------------
# Alignment & Data Processing
# ------------------------------------------------------------------------
def align_teacher_probs_to_student(teacher_logprobs, student_encodings, text_content):
    char_logprobs = [None] * len(text_content)
    curr = 0
    for t in teacher_logprobs:
        logp = t['logprob']
        token_len = len(t['token'])
        for k in range(curr, min(curr + token_len, len(text_content))):
            char_logprobs[k] = logp
        curr += token_len
    
    aligned = []
    for start, end in student_encodings.offset_mapping:
        if start == end:
            aligned.append(0.0)
            continue
        span_logprobs = [char_logprobs[k] for k in range(start, end) if k < len(char_logprobs) and char_logprobs[k] is not None]
        if not span_logprobs:
            aligned.append(0.0)
        else:
            aligned.append(np.exp(sum(span_logprobs) / len(span_logprobs)))
    return aligned

def process_data_dual_stream(samples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "teacher_probs": [], "length": []}
    
    for i in range(len(samples['source'])):
        prompt = f"<|im_start|>system\nYou are a decompiler.<|im_end|>\n<|im_start|>user\n{samples['assembly'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        if prompt_len >= MAX_SEQ_LENGTH - 10: 
            continue 

        # Stream A: SFT
        ds_tokens = tokenizer(prompt + samples['reasoning'][i], truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)
        ds_labels = list(ds_tokens["input_ids"])
        ds_labels[:min(len(ds_labels), prompt_len)] = [-100] * min(len(ds_labels), prompt_len)
        
        if not all(l == -100 for l in ds_labels):
            model_inputs["input_ids"].append(ds_tokens["input_ids"])
            model_inputs["attention_mask"].append(ds_tokens["attention_mask"])
            model_inputs["labels"].append(ds_labels)
            model_inputs["teacher_probs"].append([0.0] * len(ds_tokens["input_ids"]))
            model_inputs["length"].append(len(ds_tokens["input_ids"]))

        # Stream B: Distillation
        kimi_text = "".join([t['token'] for t in samples['logprobs'][i]])
        if kimi_text.strip():
            kimi_tokens = tokenizer(prompt + kimi_text, truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False, return_offsets_mapping=True)
            k_labels = list(kimi_tokens["input_ids"])
            mask_len = min(len(k_labels), prompt_len)
            k_labels[:mask_len] = [-100] * mask_len
            
            if not all(l == -100 for l in k_labels):
                resp_enc = tokenizer(kimi_text, return_offsets_mapping=True, add_special_tokens=False)
                aligned_probs = align_teacher_probs_to_student(samples['logprobs'][i], resp_enc, kimi_text)
                
                final_probs = [0.0] * mask_len + aligned_probs
                current_len = len(kimi_tokens["input_ids"])
                if len(final_probs) > current_len:
                    final_probs = final_probs[:current_len]
                else:
                    final_probs += [0.0] * (current_len - len(final_probs))
                
                model_inputs["input_ids"].append(kimi_tokens["input_ids"])
                model_inputs["attention_mask"].append(kimi_tokens["attention_mask"])
                model_inputs["labels"].append(k_labels)
                model_inputs["teacher_probs"].append(final_probs)
                model_inputs["length"].append(current_len)
            
    return model_inputs

def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    def collator(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in batch], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
        teacher_probs = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["teacher_probs"], dtype=torch.float32) for x in batch], batch_first=True, padding_value=0.0)

        rem = input_ids.size(1) % 64
        if rem != 0:
            pad = 64 - rem
            input_ids = F.pad(input_ids, (0, pad), value=pad_id)
            attention_mask = F.pad(attention_mask, (0, pad), value=0)
            labels = F.pad(labels, (0, pad), value=-100)
            teacher_probs = F.pad(teacher_probs, (0, pad), value=0.0)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "teacher_probs": teacher_probs}
    return collator

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    raw_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    splits = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    train_formatted = splits['train'].map(lambda x: process_data_dual_stream(x, tokenizer), batched=True, remove_columns=splits['train'].column_names, num_proc=8)
    eval_formatted = splits['test'].map(lambda x: process_data_dual_stream(x, tokenizer), batched=True, remove_columns=splits['test'].column_names, num_proc=8)
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=1, per_device_eval_batch_size=1, 
        gradient_accumulation_steps=4, learning_rate=1e-4, num_train_epochs=1, bf16=True,
        logging_steps=5, save_strategy="steps", eval_strategy="steps", save_steps=50, eval_steps=50,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="loss", greater_is_better=False,
        group_by_length=True, length_column_name="length", remove_unused_columns=False, report_to="none", dataloader_pin_memory=False
    )

    trainer = ChunkedDualStreamTrainer(model=model, args=args, train_dataset=train_formatted, eval_dataset=eval_formatted, data_collator=make_collator(tokenizer))
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # Push to HuggingFace Hub 
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        print("Pushing Model to HuggingFace Hub...")
        trainer.push_to_hub(commit_message="Initial commit", token=hf_token)
