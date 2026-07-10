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
# 1. Configuration
# ------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507" 
DATASET_FILE = "data/intermediate/all_data_new_reason5.jsonl" 
OUTPUT_DIR = "decompiler-distil-model"

# Hyperparameters
MAX_SEQ_LENGTH = 90000
ALPHA = 0.5   # Weight for Distillation Loss (Only applies to Kimi samples)
TEMP = 2.0    # Temperature for softening Kimi's probability distribution
LORA_R = 16   
LORA_ALPHA = 32

# ------------------------------------------------------------------------
# 2. Alignment Core (Solves Kimi vs Qwen Tokenizer Mismatch)
# ------------------------------------------------------------------------
def align_teacher_probs_to_student(teacher_logprobs, student_encodings, text_content):
    """
    Maps Teacher (Kimi) logprobs to Student (Qwen) tokens via Character-Level Projection.
    """
    # 1. Paint probabilities onto characters of the text
    char_probs = [0.0] * len(text_content)
    current_char_idx = 0
    
    for t_data in teacher_logprobs:
        token_str = t_data['token']
        # Convert logprob to linear probability (0-1) for averaging
        prob = np.exp(t_data['logprob']) 
        token_len = len(token_str)
        
        # Clamp to string length to prevent index errors
        end_idx = min(current_char_idx + token_len, len(text_content))
        
        for i in range(current_char_idx, end_idx):
            char_probs[i] = prob
            
        current_char_idx += token_len

    # 2. Project character probs onto Student tokens
    student_offsets = student_encodings.offset_mapping
    aligned_probs = []

    for (start, end) in student_offsets:
        if start == end: # Special token
            aligned_probs.append(0.0)
            continue
            
        # Get probs for the character span this token covers
        span_probs = char_probs[start:end]
        
        if not span_probs:
            aligned_probs.append(0.0)
        else:
            # Average confidence over the token's span
            avg_prob = sum(span_probs) / len(span_probs)
            aligned_probs.append(avg_prob)

    return aligned_probs

# ------------------------------------------------------------------------
# 3. Data Processor (Dual-Stream Splitter)
# ------------------------------------------------------------------------
def process_data_dual_stream(samples, tokenizer):
    """
    Splits one dataset entry into TWO training samples.
    """
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "teacher_probs": []}
    
    for i in range(len(samples['source'])):
        # Base Prompt Construction
        prompt = (
            "<|im_start|>system\nYou are a decompiler.<|im_end|>\n"
            f"<|im_start|>user\n{samples['assembly'][i]}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # -------------------------------------------------------
        # Pre-check: Validate Kimi data before adding anything
        # -------------------------------------------------------
        kimi_logprobs = samples['logprobs'][i]
        kimi_text = "".join([t['token'] for t in kimi_logprobs])
        
        # Skip entire sample if Kimi output is empty
        if not kimi_text.strip():
            continue

        # -------------------------------------------------------
        # STREAM A: DeepSeek (SFT Only)
        # -------------------------------------------------------
        deepseek_text = samples['reasoning'][i]
        full_text_ds = prompt + deepseek_text
        
        tokenized_ds = tokenizer(
            full_text_ds, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            padding=False,
            add_special_tokens=False  # FIX: Consistent special token handling
        )
        
        input_ids_ds = tokenized_ds["input_ids"]
        labels_ds = input_ids_ds.copy()
        # Mask prompt
        labels_ds[:prompt_len] = [-100] * min(len(labels_ds), prompt_len)
        
        # Dummy probs (0.0 means ignore KD loss)
        probs_ds = [0.0] * len(input_ids_ds)

        model_inputs["input_ids"].append(input_ids_ds)
        model_inputs["attention_mask"].append(tokenized_ds["attention_mask"])
        model_inputs["labels"].append(labels_ds)
        model_inputs["teacher_probs"].append(probs_ds)

        # -------------------------------------------------------
        # STREAM B: Kimi (Knowledge Distillation)
        # -------------------------------------------------------
        full_text_kimi = prompt + kimi_text
        
        # Tokenize with Qwen (Student) + Offset Mapping
        tokenized_kimi = tokenizer(
            full_text_kimi, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            padding=False,
            add_special_tokens=False,  # FIX: Consistent special token handling
            return_offsets_mapping=True  # Essential for alignment
        )
        
        input_ids_kimi = tokenized_kimi["input_ids"]
        labels_kimi = input_ids_kimi.copy()
        labels_kimi[:prompt_len] = [-100] * min(len(labels_kimi), prompt_len)
        
        # --- ALIGNMENT STEP ---
        # 1. Tokenize just the response to get offsets relative to the response string
        response_only_enc = tokenizer(kimi_text, return_offsets_mapping=True, add_special_tokens=False)
        
        # 2. Run Alignment
        aligned_response_probs = align_teacher_probs_to_student(
            kimi_logprobs, 
            response_only_enc, 
            kimi_text
        )
        
        # 3. Construct final prob vector
        final_probs_kimi = [0.0] * prompt_len + aligned_response_probs
        
        # 4. Truncate/Pad to match input_ids length
        final_probs_kimi = final_probs_kimi[:len(input_ids_kimi)]
        if len(final_probs_kimi) < len(input_ids_kimi):
             final_probs_kimi += [0.0] * (len(input_ids_kimi) - len(final_probs_kimi))

        model_inputs["input_ids"].append(input_ids_kimi)
        model_inputs["attention_mask"].append(tokenized_kimi["attention_mask"])
        model_inputs["labels"].append(labels_kimi)
        model_inputs["teacher_probs"].append(final_probs_kimi)
        
    return model_inputs

# ------------------------------------------------------------------------
# 4. Custom Collator & Trainer
# ------------------------------------------------------------------------
def custom_collator(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    teacher_probs = [torch.tensor(item["teacher_probs"]) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    teacher_probs = torch.nn.utils.rnn.pad_sequence(teacher_probs, batch_first=True, padding_value=0.0)
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels, 
        "teacher_probs": teacher_probs
    }

class DualStreamTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_probs = inputs.pop("teacher_probs")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 1. SFT Loss (All Samples)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        sft_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 2. KD Loss (Kimi Samples Only)
        shift_teacher_probs = teacher_probs[..., 1:].contiguous()
        kd_mask = (shift_teacher_probs > 1e-6).float()  # Only compute where we have Teacher info
        
        if kd_mask.sum() > 0:
            student_log_probs = F.log_softmax(shift_logits / TEMP, dim=-1)
            
            # FIX: Safe gather - replace -100 with 0 to avoid index errors
            gather_labels = shift_labels.clone()
            gather_labels[gather_labels < 0] = 0
            selected_log_probs = torch.gather(student_log_probs, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
            
            # Sparse KD: Minimize difference between Teacher Prob and Student LogProb
            kd_loss = -(shift_teacher_probs * selected_log_probs * kd_mask).sum() / kd_mask.sum()
        else:
            kd_loss = 0.0

        # Weighted Sum
        total_loss = ((1.0 - ALPHA) * sft_loss) + (ALPHA * kd_loss)
        
        return (total_loss, outputs) if return_outputs else total_loss

# ------------------------------------------------------------------------
# 5. Execution
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Dataset: {DATASET_FILE}")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    
    print("Processing Data (Dual Stream)...")
    formatted_dataset = dataset.map(
        lambda x: process_data_dual_stream(x, tokenizer), 
        batched=True, 
        remove_columns=dataset.column_names
    )
    print(f"Total Training Samples: {len(formatted_dataset)}")

    print(f"Loading Model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False, 
        report_to="none"
    )

    trainer = DualStreamTrainer(
        model=model,
        args=args,
        train_dataset=formatted_dataset,
        data_collator=custom_collator,
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
