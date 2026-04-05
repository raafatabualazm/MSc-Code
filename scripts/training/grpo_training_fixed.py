import torch
import re
import os
import tempfile
import subprocess
import threading
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

# --- 1. Accelerator and Model/Tokenizer Setup ---

accelerator = Accelerator()

# Note: Quantization (BnB) can be tricky with PPO and multiple models.
# We'll disable it for stability, as DDP + LoRA is already complex.

model_dir = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
# PPO requires a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model for training (with LoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# Load the reference model (frozen, no LoRA)
# This is crucial for KL divergence in PPO
ref_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
ref_model.eval()  # Set to evaluation mode

model.config.use_cache = False
ref_model.config.use_cache = False
model.config.pretraining_tp = 1
ref_model.config.pretraining_tp = 1


# --- 2. PEFT (LoRA) Configuration ---
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.08,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
)

# Apply LoRA to the main model
model = get_peft_model(model, peft_config)
accelerator.print(f"LoRA model created and ready for training:\n{model}")


# --- 2.5. Prompt Template ---
inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}. 
Please convert the following assembly code to idiomatic and clear {} code. 
You MUST use the following function signature:
```dart
{}
```

Include any necessary imports (e.g., import 'dart:math', import 'dart:core') at the beginning.
Write ONLY the function implementation - do NOT include test code or main().

### Assembly:
{}

### Response:
<think>
"""


# --- 3. Dart Sandbox Reward Function (The "Environment") ---

def run_dart_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    """
    Runs Dart code in a secure sandbox by concatenating solution and test code
    and executing it with `dart run`.
    Returns a (reward, log_message) tuple.
    """
    
    # Validate solution code
    if not solution_code.strip():
        return -1.0, "Error: Empty solution code"
    
    # Solution should NOT contain main() - that comes from test_code
    if 'void main(' in solution_code or 'main()' in solution_code:
        return -1.0, "Error: Solution should only contain the function, not main()"
    
    # Extract imports/pragmas and function code for proper file structure
    # Dart convention: imports at top, then code
    lines = solution_code.split('\n')
    imports = []
    function_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Check if line is an import, export, or pragma
        if (stripped.startswith('import ') or 
            stripped.startswith('export ') or 
            stripped.startswith('@pragma(') or
            stripped.startswith('library ') or
            stripped.startswith('part ')):
            imports.append(line)
        else:
            function_lines.append(line)
    
    # Reconstruct with proper structure: imports first, then function, then tests
    imports_section = '\n'.join(imports) if imports else ''
    function_section = '\n'.join(function_lines).strip()
    
    # Combine: imports + function + test_code
    if imports_section:
        full_code = imports_section + "\n\n" + function_section + "\n\n" + test_code
    else:
        full_code = function_section + "\n\n" + test_code
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Create a single .dart file
            test_filepath = os.path.join(temp_dir, 'temp_test.dart')
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(full_code)

            # 2. Run `dart run`
            test_proc = subprocess.run(
                ['dart', 'run', test_filepath], 
                cwd=temp_dir, 
                capture_output=True, 
                text=True, 
                timeout=timeout, 
                encoding='utf-8'
            )

            # 3. Assign rewards based on exit code
            if test_proc.returncode == 0:
                # All tests passed
                return 1.0, f"All tests passed: {test_proc.stdout[-100:]}"
            elif "Error:" in test_proc.stderr or "Error:" in test_proc.stdout:
                # Catch compilation or runtime errors
                error_msg = test_proc.stderr[:200] or test_proc.stdout[:200]
                return -1.0, f"Runtime/Compilation Error: {error_msg}"
            else:
                # Test failure (assertion/expect failed)
                error_msg = test_proc.stderr[:200] or test_proc.stdout[:200]
                return -0.5, f"Test Failure: {error_msg}"

    except subprocess.TimeoutExpired:
        return -2.0, "Timeout"
    except Exception as e:
        return -2.0, f"Sandbox Error: {str(e)}"


# --- 4. Dataset Loading and Preparation ---
accelerator.print("Loading custom JSONL dataset for Dart...")
dataset = load_dataset("json", data_files="dart_all.jsonl", split="train")

def prepare_dataset_entry(example):
    # Create the prompt from 'lang', 'assembly', and 'dart_function_signature'
    prompt_text = inference_prompt_style.format(
        example['lang'], 
        example['lang'], 
        example['dart_function_signature'],
        example['assembly']
    )
    
    # Tokenize the prompt (the query)
    tokenized = tokenizer(
        prompt_text, 
        truncation=True, 
        max_length=1024,
        padding=False
    )
    
    return {
        "input_ids": tokenized.input_ids,
        "prompt": prompt_text,
        "tests": example["tests"],
        "dart_function_signature": example.get("dart_function_signature", "")
    }

train_dataset = dataset.map(prepare_dataset_entry)
# Keep only necessary columns
train_dataset = train_dataset.remove_columns(
    [col for col in dataset.column_names if col not in ['input_ids', 'prompt', 'tests', 'dart_function_signature']]
)
train_dataset.set_format(type='torch')

def collate_fn_rl(batch):
    """Collator for RL training - passes raw dictionary items"""
    return {
        "input_ids": [item['input_ids'] for item in batch],
        "prompts": [item['prompt'] for item in batch],
        "tests": [item['tests'] for item in batch],
        "signatures": [item.get('dart_function_signature', '') for item in batch]
    }


# --- 5. PPO and GRPO Configuration ---
accelerator.print("Setting up PPO/GRPO trainer...")
K_SAMPLES = 4  # GRPO Group Size (reduced from 8 for faster iteration)

training_arguments = TrainingArguments(
    output_dir="decompiler-grpo-v1",
    num_train_epochs=4,
    remove_unused_columns=False,
)

ppo_config = PPOConfig(
    batch_size=2,        # Reduced for memory efficiency
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,  # Increased from 1.41e-6 for better learning
    kl_penalty="kl",
    kl_coeff=0.05,
    target_kl=0.1,
    ppo_epochs=4,
    log_with="none",  # Set to "wandb" or "tensorboard" if you want logging
    use_score_scaling=True,
    use_score_norm=True,
)

# Generation settings for rollouts
gen_kwargs = {
    "max_new_tokens": 1024,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
}

# --- 6. Trainer and Dataloader Initialization ---
train_dataloader = DataLoader(
    train_dataset,
    batch_size=ppo_config.batch_size,
    collate_fn=collate_fn_rl,
    shuffle=True
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# Prepare everything with Accelerator
model, ref_model, ppo_trainer, train_dataloader = accelerator.prepare(
    model, ref_model, ppo_trainer, train_dataloader
)

# --- 7. The GRPO Training Loop ---
accelerator.print(f"--- Starting GRPO Training ---")
accelerator.print(f"  Epochs: {training_arguments.num_train_epochs}")
accelerator.print(f"  Prompts per PPO Batch (batch_size): {ppo_config.batch_size}")
accelerator.print(f"  Samples per Prompt (K_SAMPLES): {K_SAMPLES}")
accelerator.print(f"  Total samples per PPO Batch: {ppo_config.batch_size * K_SAMPLES}")
accelerator.print(f"  Learning Rate: {ppo_config.learning_rate}")
accelerator.print(f"---------------------------------")

global_step = 0

for epoch in range(training_arguments.num_train_epochs):
    accelerator.print(f"\nStarting Epoch {epoch+1}/{training_arguments.num_train_epochs}")
    pbar = tqdm(enumerate(train_dataloader), disable=not accelerator.is_main_process, total=len(train_dataloader))
    
    for batch_idx, batch in pbar:
        # batch = {"input_ids": [...], "prompts": [...], "tests": [...], "signatures": [...]}
        
        # --- 7a. Rollout Phase (Generate k Samples) ---
        query_tensors = [torch.tensor(ids).to(accelerator.device) for ids in batch['input_ids']]
        
        response_tensors = []
        all_generated_texts = []
        
        for query_tensor in query_tensors:
            query_batch = query_tensor.unsqueeze(0)  # Add batch dim
            
            # Generate K_SAMPLES for this single prompt
            generated = accelerator.unwrap_model(ppo_trainer.model).generate(
                query_batch,
                num_return_sequences=K_SAMPLES,
                **gen_kwargs
            )
            
            # Decode the full texts
            decoded_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # FIXED: Extract only the response tokens, not the full sequence
            for gen_seq in generated:
                response_only = gen_seq[len(query_tensor):]
                response_tensors.append(response_only)
            
            all_generated_texts.extend(decoded_texts)

        # `queries_repeated` will have [q1, q1, q1, q1, q2, q2, q2, q2, ...]
        queries_repeated = [q for q in query_tensors for _ in range(K_SAMPLES)]
        
        # --- 7b. Reward & GRPO Advantage Calculation ---
        rewards = []
        raw_rewards_log = []
        
        example_idx = 0
        for ex_prompt, ex_tests in zip(batch['prompts'], batch['tests']):
            group_rewards = []
            
            for i in range(K_SAMPLES):
                gen_idx = example_idx * K_SAMPLES + i
                gen_text = all_generated_texts[gen_idx]
                
                # Extract just the completion
                completion = gen_text[len(ex_prompt):]
                
                # Extract Dart code from Markdown block
                solution_match = re.search(r"```dart\n(.*?)\n```", completion, re.DOTALL)
                
                if solution_match:
                    solution_code = solution_match.group(1)
                    # Get reward from sandbox
                    reward_val, log_msg = run_dart_sandbox(solution_code, ex_tests)
                else:
                    # Softer penalty for format issues (was -2.0)
                    reward_val = -0.5
                    log_msg = "Formatting Error: No ```dart block found."

                group_rewards.append(reward_val)
                raw_rewards_log.append(reward_val)
            
            # GRPO Logic: Calculate advantage relative to the group mean
            group_mean = sum(group_rewards) / len(group_rewards)
            group_advantages = [torch.tensor(r - group_mean).to(accelerator.device) for r in group_rewards]
            rewards.extend(group_advantages)
            
            example_idx += 1
        
        # --- 7c. PPO Optimization Step ---
        stats = ppo_trainer.step(queries_repeated, response_tensors, rewards)
        
        # FIXED: Log stats with proper indentation and calculation
        if accelerator.is_main_process:
            mean_raw_reward = sum(raw_rewards_log) / len(raw_rewards_log) if raw_rewards_log else 0.0
            pbar.set_description(
                f"Epoch {epoch+1} | Step {global_step} | "
                f"Mean Reward: {mean_raw_reward:.3f} | "
                f"PPO Loss: {stats.get('ppo/loss/policy', 0):.4f}"
            )
        
        global_step += 1
        
        # Periodic checkpointing
        if (batch_idx % 100 == 0 and batch_idx > 0) and accelerator.is_main_process:
            checkpoint_dir = f"{training_arguments.output_dir}/checkpoint-epoch{epoch+1}-step{global_step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            accelerator.print(f"Checkpoint saved to {checkpoint_dir}")

# --- 8. Save Final Model ---
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("\nTraining finished. Saving model...")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(training_arguments.output_dir)
    tokenizer.save_pretrained(training_arguments.output_dir)
    accelerator.print(f"Model saved to {training_arguments.output_dir}")

# --- 9. Final Inference Test ---
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("\n--- Running Final Inference Test ---")
    
    # Load a test sample (use a different index for testing)
    test_dataset = load_dataset("json", data_files="dart_all.jsonl", split="train")
    test_example = test_dataset[min(10, len(test_dataset)-1)]
    
    prompt_text = inference_prompt_style.format(
        test_example['lang'],
        test_example['lang'],
        test_example['dart_function_signature'],
        test_example['assembly']
    )
    
    inputs = tokenizer([prompt_text], return_tensors="pt").to(accelerator.device)
    
    # Use the final trained model (unwrapped)
    final_model = accelerator.unwrap_model(model)
    final_model.eval()
    
    with torch.no_grad():
        outputs = final_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=0.1,  # Low temp for best guess
            top_p=1.0,
        )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print(f"PROMPT (Test Sample):")
    print(f"{'='*80}")
    print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
    print(f"\n{'='*80}")
    print(f"GENERATED RESPONSE:")
    print(f"{'='*80}")
    completion_text = response[0][len(prompt_text):]
    print(completion_text)
    
    # Run final check
    print(f"\n{'='*80}")
    print(f"RUNNING FINAL TEST:")
    print(f"{'='*80}")
    
    solution_match = re.search(r"```dart\n(.*?)\n```", completion_text, re.DOTALL)
    
    if solution_match:
        solution_code = solution_match.group(1)
        test_code = test_example['tests']
        reward, log = run_dart_sandbox(solution_code, test_code)
        print(f"Reward: {reward}")
        print(f"Log: {log}")
    else:
        print("Final test failed: Could not parse ```dart code block from response.")

accelerator.print("\n✅ Training pipeline completed successfully!")
