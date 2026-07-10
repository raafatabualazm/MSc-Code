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
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import gc

# ============================================================================
# H200 DoRA-OPTIMIZED GRPO TRAINING SCRIPT
# Hardware: NVIDIA H200 (141GB HBM3e)
# Features: DoRA enabled + Long sequences (2048 or 3072)
# Strategy: Smaller batches, aggressive memory management
# ============================================================================

print("🚀 H200 DoRA-Optimized GRPO Training Script")
print("=" * 80)

# CONFIGURATION SELECTOR
# ========================================================================
# Choose your configuration here:
SEQUENCE_MODE = "2048"  # Options: "2048" or "3072"
# ========================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

accelerator = Accelerator(mixed_precision="bf16")

model_dir = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

accelerator.print("=" * 80)
accelerator.print("CONFIGURATION")
accelerator.print("=" * 80)
accelerator.print(f"  Device: {accelerator.device}")
accelerator.print(f"  Mixed Precision: bf16")
accelerator.print(f"  Available Memory: ~141GB (H200)")
accelerator.print(f"  Sequence Mode: {SEQUENCE_MODE} tokens")
accelerator.print(f"  DoRA: ENABLED")
accelerator.print(f"  Memory Optimization: AGGRESSIVE")
accelerator.print("=" * 80)

# --- PEFT Configuration with DoRA ---
# Using DoRA for better performance, but with smaller rank to fit memory
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,                 # Keep rank modest with DoRA
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,        # ✓ DoRA ENABLED as requested
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

accelerator.print("\nLoading models with DoRA...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    peft_config=peft_config
)

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

# CRITICAL: Enable gradient checkpointing
model.pretrained_model.gradient_checkpointing_enable()

accelerator.print("✓ Models loaded with DoRA")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
accelerator.print(f"✓ DoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")

# --- Prompt Template ---
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

# --- Dart Sandbox Reward Function ---
def run_dart_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    if not solution_code.strip():
        return -1.0, "Error: Empty solution code"
    
    if 'void main(' in solution_code or 'main()' in solution_code:
        return -1.0, "Error: Solution should only contain the function, not main()"
    
    lines = solution_code.split('\n')
    imports = []
    function_lines = []
    
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith('import ') or 
            stripped.startswith('export ') or 
            stripped.startswith('@pragma(') or
            stripped.startswith('library ') or
            stripped.startswith('part ')):
            imports.append(line)
        else:
            function_lines.append(line)
    
    imports_section = '\n'.join(imports) if imports else ''
    function_section = '\n'.join(function_lines).strip()
    
    if imports_section:
        full_code = imports_section + "\n\n" + function_section + "\n\n" + test_code
    else:
        full_code = function_section + "\n\n" + test_code
    
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

# --- Dataset Loading ---
accelerator.print("\nLoading dataset...")
dataset = load_dataset("json", data_files="data/testing/grpo_data.jsonl", split="train")
accelerator.print(f"✓ Loaded {len(dataset):,} training examples")

def prepare_dataset_entry(example):
    prompt_text = inference_prompt_style.format(
        example['lang'], 
        example['lang'], 
        example['dart_function_signature'],
        example['assembly']
    )
    return {
        'prompts': prompt_text,
        'tests': example['tests']
    }

dataset = dataset.map(prepare_dataset_entry)

# --- Configuration Based on Sequence Length ---
EPOCHS = 3
LEARNING_RATE = 1e-5

if SEQUENCE_MODE == "2048":
    # Configuration for 2048 tokens
    # Estimated memory: ~95-105 GB
    accelerator.print("\n⚙️  Using 2048 token configuration")
    PROMPTS_PER_BATCH = 2
    K_SAMPLES = 6
    MAX_LENGTH = 2048
    MAX_NEW_TOKENS = 1536
    GRAD_ACCUM_STEPS = 3  # Effective batch = 6 prompts
    MINI_BATCH_SIZE = 2
    accelerator.print(f"   • Prompts/batch: {PROMPTS_PER_BATCH}")
    accelerator.print(f"   • Samples/prompt: {K_SAMPLES}")
    accelerator.print(f"   • Total sequences: {PROMPTS_PER_BATCH * K_SAMPLES}")
    accelerator.print(f"   • Expected memory: 95-105 GB")
    
elif SEQUENCE_MODE == "3072":
    # Configuration for 3072 tokens (VERY AGGRESSIVE)
    # Estimated memory: ~115-125 GB
    accelerator.print("\n⚙️  Using 3072 token configuration (AGGRESSIVE)")
    PROMPTS_PER_BATCH = 2
    K_SAMPLES = 4
    MAX_LENGTH = 3072
    MAX_NEW_TOKENS = 2048
    GRAD_ACCUM_STEPS = 4  # Effective batch = 8 prompts
    MINI_BATCH_SIZE = 2
    accelerator.print(f"   • Prompts/batch: {PROMPTS_PER_BATCH}")
    accelerator.print(f"   • Samples/prompt: {K_SAMPLES}")
    accelerator.print(f"   • Total sequences: {PROMPTS_PER_BATCH * K_SAMPLES}")
    accelerator.print(f"   • Expected memory: 115-125 GB")
    accelerator.print(f"   • ⚠️  WARNING: Pushing memory limits!")
else:
    raise ValueError(f"Invalid SEQUENCE_MODE: {SEQUENCE_MODE}")

accelerator.print("=" * 80)

training_arguments = TrainingArguments(
    output_dir="./dart_grpo_dora_h200",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PROMPTS_PER_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    optim="adamw_torch",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to="none"
)

# --- PPO Configuration ---
TOTAL_SEQUENCES = PROMPTS_PER_BATCH * K_SAMPLES

ppo_config = PPOConfig(
    batch_size=TOTAL_SEQUENCES,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    log_with=None,
    ppo_epochs=3,
    early_stopping=False,
    target_kl=0.1,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    gamma=1.0,
    lam=0.95,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# --- Main Training Loop ---
accelerator.print(f"\n{'='*80}")
accelerator.print("STARTING DoRA TRAINING WITH LONG SEQUENCES")
accelerator.print(f"{'='*80}")
accelerator.print(f"Total Epochs: {EPOCHS}")
accelerator.print(f"Sequence length: {MAX_LENGTH}")
accelerator.print(f"Max new tokens: {MAX_NEW_TOKENS}")
accelerator.print(f"Batch: {PROMPTS_PER_BATCH} prompts × {K_SAMPLES} samples = {TOTAL_SEQUENCES} sequences")
accelerator.print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} (effective: {PROMPTS_PER_BATCH * GRAD_ACCUM_STEPS} prompts)")
accelerator.print(f"{'='*80}\n")

dataloader = DataLoader(
    dataset, 
    batch_size=training_arguments.per_device_train_batch_size,
    shuffle=True,
    collate_fn=lambda x: {
        'prompts': [item['prompts'] for item in x],
        'tests': [item['tests'] for item in x]
    }
)

gen_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

best_reward = float('-inf')
global_step = 0

for epoch in range(EPOCHS):
    if accelerator.is_main_process:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    else:
        pbar = dataloader
    
    epoch_rewards = []
    
    for batch_idx, batch in enumerate(pbar):
        # Aggressive memory clearing before batch
        torch.cuda.empty_cache()
        gc.collect()
        
        # --- Generation Phase ---
        query_tensors = []
        for prompt_text in batch['prompts']:
            tokens = tokenizer(
                prompt_text, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True, 
                max_length=MAX_LENGTH
            )
            query_tensors.append(tokens.input_ids.squeeze(0).to(accelerator.device))
        
        response_tensors = []
        all_generated_texts = []
        
        # Generate sequences one prompt at a time for better memory control
        for query_idx, query_tensor in enumerate(query_tensors):
            query_batch = query_tensor.unsqueeze(0)
            
            with torch.no_grad():
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
            
            # Critical: Clear immediately after each prompt's generation
            del generated
            torch.cuda.empty_cache()
            
            # Log memory after each prompt (only for first batch)
            if batch_idx == 0 and accelerator.is_main_process:
                allocated = torch.cuda.memory_allocated() / 1e9
                accelerator.print(f"   Prompt {query_idx+1}/{len(query_tensors)} generated - Memory: {allocated:.1f}GB")

        # Create repeated queries list
        queries_repeated = [
            q.clone() 
            for q in query_tensors 
            for _ in range(K_SAMPLES)
        ]
        
        # --- Reward & GRPO Advantage Calculation ---
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
            
            # GRPO: Calculate advantage
            group_mean = sum(group_rewards) / len(group_rewards)
            group_advantages = [
                torch.tensor(r - group_mean, dtype=torch.float32).to(accelerator.device) 
                for r in group_rewards
            ]
            rewards.extend(group_advantages)
            
            example_idx += 1
        
        # --- PPO Optimization Step ---
        stats = ppo_trainer.step(queries_repeated, response_tensors, rewards)
        
        # Aggressive cleanup after PPO step
        del queries_repeated, response_tensors, rewards, query_tensors, all_generated_texts
        torch.cuda.empty_cache()
        gc.collect()
        
        # Track metrics
        mean_raw_reward = sum(raw_rewards_log) / len(raw_rewards_log) if raw_rewards_log else 0.0
        epoch_rewards.extend(raw_rewards_log)
        success_rate = successful_completions / len(raw_rewards_log) if raw_rewards_log else 0.0
        
        if accelerator.is_main_process:
            # Monitor memory closely
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            pbar.set_description(
                f"E{epoch+1} | S{global_step} | "
                f"R:{mean_raw_reward:.3f} | "
                f"Acc:{success_rate*100:.1f}% | "
                f"Mem:{allocated:.0f}GB(max:{max_allocated:.0f}GB)"
            )
            
            # Reset max memory tracker every 10 steps
            if global_step % 10 == 0:
                torch.cuda.reset_peak_memory_stats()
        
        global_step += 1
        
        # Checkpoint every 50 steps
        if (batch_idx % 50 == 0 and batch_idx > 0) and accelerator.is_main_process:
            checkpoint_dir = f"{training_arguments.output_dir}/checkpoint-epoch{epoch+1}-step{global_step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            with open(f"{checkpoint_dir}/metrics.txt", "w") as f:
                f.write(f"Global Step: {global_step}\n")
                f.write(f"Mean Reward: {mean_raw_reward:.4f}\n")
                f.write(f"Success Rate: {success_rate*100:.2f}%\n")
                f.write(f"Sequence Length: {MAX_LENGTH}\n")
                f.write(f"Max Memory: {max_allocated:.2f}GB\n")
            
            accelerator.print(f"\n💾 Checkpoint saved: {checkpoint_dir}")
    
    # End of epoch
    if accelerator.is_main_process:
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        epoch_success_rate = sum(1 for r in epoch_rewards if r == 1.0) / len(epoch_rewards) if epoch_rewards else 0.0
        
        accelerator.print(f"\n{'='*80}")
        accelerator.print(f"EPOCH {epoch+1} SUMMARY")
        accelerator.print(f"{'='*80}")
        accelerator.print(f"Mean Reward:   {epoch_mean_reward:.4f}")
        accelerator.print(f"Success Rate:   {epoch_success_rate*100:.2f}%")
        accelerator.print(f"Total Samples:  {len(epoch_rewards):,}")
        accelerator.print(f"Peak Memory:    {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        accelerator.print(f"{'='*80}\n")
        
        if epoch_mean_reward > best_reward:
            best_reward = epoch_mean_reward
            best_model_dir = f"{training_arguments.output_dir}/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            accelerator.print(f"🏆 New best model saved: {best_model_dir} (reward: {best_reward:.4f})")

# --- Save Final Model ---
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
    accelerator.print(f"  Peak memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

accelerator.print("\n" + "="*80)
accelerator.print("🎉 DoRA TRAINING COMPLETED!")
accelerator.print("="*80)


