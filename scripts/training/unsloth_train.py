import psutil
from unsloth import FastLanguageModel
import torch
import os
# Unsloth handles quantization internally - much simpler setup
max_seq_length = 90112  # Adjust based on your data
dtype = None  # Auto-detect (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)
load_in_4bit = True  # Use 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Thinking-2507",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# Apply LoRA/DoRA with Unsloth - automatically targets optimal modules
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.07,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
    random_state=3407,
    use_rslora=False,
    use_dora=True,  # Enable DoRA
    loftq_config=None,
)

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
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

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["assembly"]
    outputs = examples["source"]
    langs = examples["language"]
    reasonings = examples["reasoning"]
    texts = []
    for assembly, source, lang, reasoning in zip(inputs, outputs, langs, reasonings):
        if not source.endswith(EOS_TOKEN):
            source += EOS_TOKEN
        text = train_prompt_style.format(lang, lang, assembly, reasoning, lang, source)
        texts.append(text)
    return {"text": texts}


from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/intermediate/all_data_new_reason5.jsonl",
    split="train"
)

dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

# Pre-training inference test
inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
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

assembly = dataset[237]['assembly']
lang = dataset[237]['language']



# Enable inference mode for generation
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [inference_prompt_style.format(lang, lang, assembly)],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=16384,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    temperature=0.16,
    top_p=1.0,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("=== Pre-training output ===")
print(response[0].split("### Response:")[1])

# Switch back to training mode
FastLanguageModel.for_training(model)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output-v7-unsloth",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",  # Unsloth optimized 8-bit AdamW
    num_train_epochs=4,
    logging_steps=0.1,
    warmup_ratio=0.1,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    group_by_length=True,
    report_to="none",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=3407,
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=False,  # Set True for shorter sequences to speed up training
)

import gc
gc.collect()
torch.cuda.empty_cache()

# Train
trainer_stats = trainer.train()

# Post-training inference
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [inference_prompt_style.format(lang, lang, assembly)],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=8192,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    temperature=0.16,
    top_p=1.0,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("\n=== Post-training output ===")
print(response[0].split("### Response:")[1])

source = dataset[237]['source']
print()
print("=======================================================")
print("=== Ground Truth ===")
print(source)

# Save the model
model.save_pretrained("output-v7-unsloth/final_model")
tokenizer.save_pretrained("output-v7-unsloth/final_model")

# Optional: Save merged 16-bit model for deployment
# model.save_pretrained_merged("output-v7-unsloth/merged_16bit", tokenizer, save_method="merged_16bit")

# Optional: Push to HuggingFace Hub
hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
if hf_token is not None:
    print("Pushing Model to HuggingFace Hub...")
    model.push_to_hub_merged("raafatabualazm/decompiler-single-v1", tokenizer, save_method="merged_16bit", token=hf_token)

