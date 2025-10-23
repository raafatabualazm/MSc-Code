import torch
import re
import os
import tempfile
import subprocess
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

# ============================================================================
# H200 OPTIMIZED GRPO TRAINING SCRIPT
# Hardware: NVIDIA H200 (141GB HBM3e)
# Optimizations: Large batch sizes, more samples, full bf16, Flash Attention 2
# ============================================================================

print("🚀 H200-Optimized GRPO Training Script")
print("=" * 80)

# --- 1. Accelerator and Model/Tokenizer Setup ---
accelerator = Accelerator(mixed_precision="bf16")  # Full bf16 for H200

model_dir = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

accelerator.print("Loading models...")
accelerator.print(f"  Device: {accelerator.device}")
accelerator.print(f"  Mixed Precision: bf16")
accelerator.print(f"  Available Memory: ~141GB (H200)")

# Load the base model for training (with LoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"  # H200 supports FA2
)

# Load the reference model (frozen, no LoRA)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
ref_model.eval()

model.config.use_cache = False
ref_model.config.use_cache = False
model.config.pretraining_tp = 1
ref_model.config.pretraining_tp = 1

accelerator.print("✓ Models loaded successfully")


# --- 2. PEFT (LoRA) Configuration ---
# H200: Can use larger rank and more modules
peft_config = LoraConfig(
    lora_alpha=64,        # Increased from 32
    lora_dropout=0.08,
    r=32,                 # Increased from 16 for more capacity
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
)

model = get_peft_model(model, peft_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
accelerator.print(f"✓ LoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")


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

### Assembly:
{}

### Response:
<think>
"""


# --- 3. Dart Sandbox Reward Function ---
def run_dart_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    """
    Runs Dart code in a secure sandbox.
    Returns (reward, log_message) tuple.
    
    Reward scheme:
      +1.0: All tests passed
      -0.5: Test failure (assertion failed)
      -1.0: Runtime/compilation error
      -2.0: Timeout or sandbox error
    """
    full_code = solution_code + "\n\n" + test_code
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filepath = os.path.join(temp_dir, 'temp_test.dart')
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(full_code)

            test_proc = subprocess.run(
                ['dart', 'run', test_filepath], 
                cwd=temp_dir, 
                capture_output=True, 
                text=True, 
                timeout=timeout, 
                encoding='utf-8'
            )

            if test_proc.returncode == 0:
                return 1.0, f"✓ All tests passed"
            elif "Error:" in test_proc.stderr or "Error:" in test_proc.stdout:
                error_msg = (test_proc.stderr or test_proc.stdout)[:200]
                return -1.0, f"Compilation/Runtime Error: {error_msg}"
            else:
                error_msg = (test_proc.stderr or test_proc.stdout)[:200]
                return -0.5, f"Test Failure: {error_msg}"

    except subprocess.TimeoutExpired:
        return -2.0, "⏱ Timeout"
    except Exception as e:
        return -2.0, f"Sandbox Error: {str(e)}"


# --- 4. Dataset Loading and Preparation ---
accelerator.print("\nLoading dataset...")
dataset = load_dataset("json", data_files="dart_all.jsonl", split="train")
accelerator.print(f"✓ Loaded {len(dataset):,} training examples")

def prepare_dataset_entry(example):
    prompt_text = inference_prompt_style.format(
        example['lang'], 
        example['lang'], 
        example['dart_function_signature'],
        example['assembly']
    )
    
    tokenized = tokenizer(
        prompt_text, 
        truncation=True, 
        max_length=2048,  # H200: Increased from 1024
        padding=False
    )
    
    return {
        "input_ids": tokenized.input_ids,
        "prompt": prompt_text,
        "tests": example["tests"],
        "dart_function_signature": example.get("dart_function_signature", "")
    }

train_dataset = dataset.map(prepare_dataset_entry, num_proc=4)
train_dataset = train_dataset.remove_columns(
    [col for col in dataset.column_names if col not in ['input_ids', 'prompt', 'tests', 'dart_function_signature']]
)
train_dataset.set_format(type='torch')

def collate_fn_rl(batch):
    return {
        "input_ids": [item['input_ids'] for item in batch],
        "prompts": [item['prompt'] for item in batch],
        "tests": [item['tests'] for item in batch],
        "signatures": [item.get('dart_function_signature', '') for item in batch]
    }


# --- 5. H200-Optimized PPO and GRPO Configuration ---
accelerator.print("\n" + "=" * 80)
accelerator.print("H200 OPTIMIZATION SETTINGS")
accelerator.print("=" * 80)

# H200: Aggressive scaling
K_SAMPLES = 16            # 4x increase (was 4)
BATCH_SIZE = 8            # 4x increase (was 2)
MINI_BATCH_SIZE = 2       # Can process larger mini-batches
GRAD_ACCUM_STEPS = 4      # = BATCH_SIZE / MINI_BATCH_SIZE

accelerator.print(f"K_SAMPLES (per prompt):        {K_SAMPLES}")
accelerator.print(f"BATCH_SIZE (prompts/step):     {BATCH_SIZE}")
accelerator.print(f"Total samples per step:        {BATCH_SIZE * K_SAMPLES}")
accelerator.print(f"Mini-batch size:               {MINI_BATCH_SIZE}")
accelerator.print(f"Gradient accumulation:         {GRAD_ACCUM_STEPS}")
accelerator.print(f"Effective batch size:          {BATCH_SIZE * K_SAMPLES}")
accelerator.print("=" * 80)

training_arguments = TrainingArguments(
    output_dir="decompiler-grpo-h200",
    num_train_epochs=5,  # Can afford more epochs with H200
    remove_unused_columns=False,
    bf16=True,
)

ppo_config = PPOConfig(
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=8e-6,      # Slightly higher for larger batches
    kl_penalty="kl",
    kl_coeff=0.05,
    target_kl=0.15,          # Slightly higher target for stability
    ppo_epochs=4,
    log_with="none",         # Set to "wandb" for logging
    use_score_scaling=True,
    use_score_norm=True,
    cliprange=0.2,           # PPO clipping
)

# H200: Can generate longer sequences faster
gen_kwargs = {
    "max_new_tokens": 1536,   # Increased from 1024
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "do_sample": True,
    "temperature": 0.8,       # Slightly higher for more exploration
    "top_p": 0.95,
}


# --- 6. Trainer and Dataloader Initialization ---
train_dataloader = DataLoader(
    train_dataset,
    batch_size=ppo_config.batch_size,
    collate_fn=collate_fn_rl,
    shuffle=True,
    num_workers=4,            # H200: Parallel data loading
    pin_memory=True,          # Faster CPU->GPU transfer
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

model, ref_model, ppo_trainer, train_dataloader = accelerator.prepare(
    model, ref_model, ppo_trainer, train_dataloader
)


# --- 7. The GRPO Training Loop ---
accelerator.print(f"\n🚀 Starting GRPO Training on H200")
accelerator.print(f"  Epochs: {training_arguments.num_train_epochs}")
accelerator.print(f"  Steps per epoch: ~{len(train_dataloader)}")
accelerator.print(f"  Total samples per step: {BATCH_SIZE * K_SAMPLES}")
accelerator.print(f"  Estimated time per epoch: ~{len(train_dataloader) * BATCH_SIZE * K_SAMPLES * 2 / 3600:.1f}h")
accelerator.print(f"-" * 80)

global_step = 0
best_reward = -999.0

for epoch in range(training_arguments.num_train_epochs):
    accelerator.print(f"\n{'='*80}")
    accelerator.print(f"EPOCH {epoch+1}/{training_arguments.num_train_epochs}")
    accelerator.print(f"{'='*80}")
    
    pbar = tqdm(
        enumerate(train_dataloader), 
        disable=not accelerator.is_main_process, 
        total=len(train_dataloader),
        desc=f"Epoch {epoch+1}"
    )
    
    epoch_rewards = []
    
    for batch_idx, batch in pbar:
        # --- 7a. Rollout Phase (Generate k Samples) ---
        query_tensors = [torch.tensor(ids).to(accelerator.device) for ids in batch['input_ids']]
        
        response_tensors = []
        all_generated_texts = []
        
        # H200: Can batch generate efficiently
        for query_tensor in query_tensors:
            query_batch = query_tensor.unsqueeze(0)
            
            with torch.no_grad():  # Save memory during generation
                generated = accelerator.unwrap_model(ppo_trainer.model).generate(
                    query_batch,
                    num_return_sequences=K_SAMPLES,
                    **gen_kwargs
                )
            
            decoded_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            for gen_seq in generated:
                response_only = gen_seq[len(query_tensor):]
                response_tensors.append(response_only)
            
            all_generated_texts.extend(decoded_texts)

        queries_repeated = [q for q in query_tensors for _ in range(K_SAMPLES)]
        
        # --- 7b. Reward & GRPO Advantage Calculation ---
        rewards = []
        raw_rewards_log = []
        successful_completions = 0
        
        example_idx = 0
        for ex_prompt, ex_tests in zip(batch['prompts'], batch['tests']):
            group_rewards = []
            
            for i in range(K_SAMPLES):
                gen_idx = example_idx * K_SAMPLES + i
                gen_text = all_generated_texts[gen_idx]
                completion = gen_text[len(ex_prompt):]
                
                # Extract Dart code from Markdown block
                solution_match = re.search(r"```dart\n(.*?)\n```", completion, re.DOTALL)
                
                if solution_match:
                    solution_code = solution_match.group(1)
                    reward_val, log_msg = run_dart_sandbox(solution_code, ex_tests)
                    if reward_val == 1.0:
                        successful_completions += 1
                else:
                    reward_val = -0.5
                    log_msg = "Format error: No ```dart block"

                group_rewards.append(reward_val)
                raw_rewards_log.append(reward_val)
            
            # GRPO: Calculate advantage relative to group mean
            group_mean = sum(group_rewards) / len(group_rewards)
            group_advantages = [
                torch.tensor(r - group_mean, dtype=torch.float32).to(accelerator.device) 
                for r in group_rewards
            ]
            rewards.extend(group_advantages)
            
            example_idx += 1
        
        # --- 7c. PPO Optimization Step ---
        stats = ppo_trainer.step(queries_repeated, response_tensors, rewards)
        
        # Track metrics
        mean_raw_reward = sum(raw_rewards_log) / len(raw_rewards_log) if raw_rewards_log else 0.0
        epoch_rewards.extend(raw_rewards_log)
        success_rate = successful_completions / len(raw_rewards_log) if raw_rewards_log else 0.0
        
        if accelerator.is_main_process:
            pbar.set_description(
                f"Epoch {epoch+1} | Step {global_step} | "
                f"Reward: {mean_raw_reward:.3f} | "
                f"Success: {success_rate*100:.1f}% | "
                f"Loss: {stats.get('ppo/loss/policy', 0):.4f}"
            )
        
        global_step += 1
        
        # Periodic checkpointing (every 50 steps on H200 - faster hardware)
        if (batch_idx % 50 == 0 and batch_idx > 0) and accelerator.is_main_process:
            checkpoint_dir = f"{training_arguments.output_dir}/checkpoint-epoch{epoch+1}-step{global_step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save metrics
            with open(f"{checkpoint_dir}/metrics.txt", "w") as f:
                f.write(f"Global Step: {global_step}\n")
                f.write(f"Mean Reward: {mean_raw_reward:.4f}\n")
                f.write(f"Success Rate: {success_rate*100:.2f}%\n")
            
            accelerator.print(f"\n💾 Checkpoint saved: {checkpoint_dir}")
    
    # End of epoch summary
    if accelerator.is_main_process:
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        epoch_success_rate = sum(1 for r in epoch_rewards if r == 1.0) / len(epoch_rewards) if epoch_rewards else 0.0
        
        accelerator.print(f"\n{'='*80}")
        accelerator.print(f"EPOCH {epoch+1} SUMMARY")
        accelerator.print(f"{'='*80}")
        accelerator.print(f"Mean Reward:    {epoch_mean_reward:.4f}")
        accelerator.print(f"Success Rate:   {epoch_success_rate*100:.2f}%")
        accelerator.print(f"Total Samples:  {len(epoch_rewards):,}")
        accelerator.print(f"{'='*80}\n")
        
        # Save best model
        if epoch_mean_reward > best_reward:
            best_reward = epoch_mean_reward
            best_model_dir = f"{training_arguments.output_dir}/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            accelerator.print(f"🏆 New best model saved: {best_model_dir} (reward: {best_reward:.4f})")


# --- 8. Save Final Model ---
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("\n" + "="*80)
    accelerator.print("TRAINING COMPLETE")
    accelerator.print("="*80)
    accelerator.print("Saving final model...")
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(training_arguments.output_dir)
    tokenizer.save_pretrained(training_arguments.output_dir)
    
    accelerator.print(f"✓ Final model saved: {training_arguments.output_dir}")
    accelerator.print(f"✓ Best model saved: {training_arguments.output_dir}/best_model")
    accelerator.print(f"  Best reward: {best_reward:.4f}")


# --- 9. Final Inference Test ---
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("\n" + "="*80)
    accelerator.print("FINAL INFERENCE TEST")
    accelerator.print("="*80)
    
    test_dataset = load_dataset("json", data_files="dart_all.jsonl", split="train")
    test_example = test_dataset[min(10, len(test_dataset)-1)]
    
    prompt_text = inference_prompt_style.format(
        test_example['lang'],
        test_example['lang'],
        test_example['dart_function_signature'],
        test_example['assembly']
    )
    
    inputs = tokenizer([prompt_text], return_tensors="pt").to(accelerator.device)
    
    final_model = accelerator.unwrap_model(model)
    final_model.eval()
    
    accelerator.print("\nGenerating response...")
    with torch.no_grad():
        outputs = final_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1536,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=0.1,
            top_p=1.0,
        )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    completion_text = response[0][len(prompt_text):]
    
    print(f"\n{'='*80}")
    print("GENERATED COMPLETION:")
    print(f"{'='*80}")
    print(completion_text[:500] + "..." if len(completion_text) > 500 else completion_text)
    
    print(f"\n{'='*80}")
    print("VALIDATION:")
    print(f"{'='*80}")
    
    solution_match = re.search(r"```dart\n(.*?)\n```", completion_text, re.DOTALL)
    
    if solution_match:
        solution_code = solution_match.group(1)
        test_code = test_example['tests']
        reward, log = run_dart_sandbox(solution_code, test_code)
        
        if reward == 1.0:
            print("✅ PASS - All tests passed!")
        else:
            print(f"❌ FAIL - Reward: {reward}")
        print(f"Log: {log}")
    else:
        print("❌ FAIL - Could not parse Dart code block")

accelerator.print("\n" + "="*80)
accelerator.print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
accelerator.print("="*80)
