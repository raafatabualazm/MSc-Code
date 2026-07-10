from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_dir = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config            
)

model.config.use_cache = False
model.config.pretraining_tp = 1

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and dart. 
Please convert the following assembly code to idiomatic and clear dart code. 

### Assembly:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["assembly"]
    outputs = examples["source"]
    texts = []
    for assembly, source in zip(inputs, outputs):
        # Append the EOS token to the response if it's not already there
        if not source.endswith(tokenizer.eos_token):
            source += tokenizer.eos_token
        text = train_prompt_style.format(assembly, source)
        texts.append(text)
    return {"text": texts}


from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="train5.jsonl",
    split="train"
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and dart. 
Please convert the following assembly code to idiomatic and clear dart code. 

### Assembly:
{}

### Response:
<think>
"""

assembly = dataset[236]['assembly']

inputs = tokenizer(
    [inference_prompt_style.format(assembly) + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")


outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    temperature=0.16,
    top_p=1.0,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0].split("### Response:")[1])


from peft import LoraConfig, get_peft_model

# LoRA config
peft_config = LoraConfig(
    lora_alpha=128,                           # Scaling factor for LoRA
    lora_dropout=0.3,                       # Add slight dropout for regularization
    r=128,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_proj",
    ],  # Target modules for LoRA
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model = get_peft_model(model, peft_config)


from trl import SFTTrainer
from transformers import TrainingArguments


# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output-v7",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    num_train_epochs=4,
    logging_steps=0.1,
    warmup_ratio=0.1,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none",
    lr_scheduler_type="cosine",
    weight_decay=0.01
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator,
)

import gc, torch
gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()



inputs = tokenizer(
    [inference_prompt_style.format(assembly) + tokenizer.eos_token],
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
print(response[0].split("### Response:")[1])

source = dataset[236]['source']

print()
print("=======================================================")
print(source)