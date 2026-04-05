from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import Accelerator  # Add this

accelerator = Accelerator()  # Add this: Initializes Accelerate state

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model_dir = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},  
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
    # quantization_config=bnb_config            
)

model.config.use_cache = False
model.config.pretraining_tp = 1

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}. 
Please convert the following assembly code to idiomatic and clear {} code. 

### Assembly:
{}

### Response:
```{}
{}
```
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["assembly"]
    outputs = examples["source"]
    langs = examples["language"]
    texts = []
    for assembly, source, lang in zip(inputs, outputs, langs):
        # Append the EOS token to the response if it's not already there
        if not source.endswith(tokenizer.eos_token):
            source += tokenizer.eos_token
        text = train_prompt_style.format(lang, lang, assembly, lang.lower(), source)
        texts.append(text)
    return {"text": texts}


from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="dart_all.jsonl",
    split="train"
)

dataset_split = dataset.train_test_split(
    test_size=0.2  # Use 10% of the data for testing
)

train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]



train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
)

test_dataset = test_dataset.map(
    formatting_prompts_func,
    batched=True,
)

inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}. 
Please convert the following assembly code to idiomatic and clear {} code. 

### Assembly:
{}

### Response:
"""

assembly = dataset[236]['assembly']
lang = dataset[236]['language']

inputs = tokenizer(
    [inference_prompt_style.format(lang, lang, assembly) + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")


outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=6000,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    temperature=0.2,
    top_p=0.99,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0].split("### Response:")[1])


from peft import LoraConfig, get_peft_model

# LoRA config
peft_config = LoraConfig(
    lora_alpha=32,                           # Scaling factor for LoRA
    lora_dropout=0.08,                       # Add slight dropout for regularization
    r=16,                                   # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    use_dora=True,                        # Use DORA (Weight-Decomposed Low-Rank Adaptation)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"
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
    output_dir="decompiler-v1",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    num_train_epochs=4,
    logging_steps=10,
    warmup_ratio=0.1,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=15,  # Evaluate every 100 steps
    save_strategy="steps",
    save_steps=30,  # Save every 100 steps
    save_total_limit=3,  # Keep only the last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    ddp_find_unused_parameters=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    data_collator=data_collator
)

import gc, torch
gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()

trainer.save_model()


inputs = tokenizer(
    [inference_prompt_style.format(lang, lang, assembly) + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")


outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=6000,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    temperature=0.2,
    top_p=0.99,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0].split("### Response:")[1])

source = dataset[236]['source']

print()
print("=======================================================")
print(source)