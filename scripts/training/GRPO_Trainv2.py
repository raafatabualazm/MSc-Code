import torch
import re
import os
import tempfile
import subprocess
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from accelerate import Accelerator
import gc
from scipy.stats import rankdata

# ============================================================================
# H200 DoRA-GRPO TRAINING SCRIPT - TRUE PER-TEST REWARD OPTIMIZATION
# ============================================================================

print("🚀 H200 DoRA-GRPO Training - True Per-Test Reward Optimization")
print("=" * 80)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CONFIGURATION SELECTOR
SEQUENCE_MODE = "2048"  # Options: "2048" or "3072"
GRPO_BONUS_WEIGHT = 0.1  # Weight for the rank-based bonus

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
accelerator.print(f"  Sequence Mode: {SEQUENCE_MODE} tokens")
accelerator.print(f"  DoRA: ENABLED")
accelerator.print(f"  GRPO: ENABLED (Ranked Bonus Weight: {GRPO_BONUS_WEIGHT})")
accelerator.print(f"  Reward System: TRUE Per-Test Reward (OPTIMIZED + STRUCTURAL GUARDS)")
accelerator.print("=" * 80)

# ============================================================================
# TRUE PER-TEST REWARD SYSTEM WITH STRUCTURAL GUARDS
# ============================================================================

def run_dart_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    """Base sandbox execution function"""
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
    full_code = (imports_section + "\n\n" if imports_section else "") + function_section + "\n\n" + test_code
    
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
                timeout=timeout
            )
            
            if test_proc.returncode == 0:
                return 1.0, "✓ All tests passed"
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

class TruePerTestReward:
    def __init__(self,
                 base_pass_reward=1.5,
                 base_fail_penalty=-0.5,
                 enable_asserts: bool = False,
                 main_violation_penalty: float = -5.0,
                 empty_code_penalty: float = -3.0):
        self.base_pass_reward = base_pass_reward
        self.base_fail_penalty = base_fail_penalty
        self.enable_asserts = enable_asserts
        self.main_violation_penalty = main_violation_penalty
        self.empty_code_penalty = empty_code_penalty

        self.precise_pattern = re.compile(r'''
            ^\s*expect\s*\(\s*candidate\s*\(\s*
            " (?P<input> (?:[^"\\]|\\.)* ) "
            \s*\)\s*,\s*
            \[ (?P<expected> (?:\s*" (?:[^"\\]|\\.)* "\s*,?)* ) \]
            \s*\)\s*;\s*$
        ''', re.MULTILINE | re.VERBOSE)

    # --- NEW: robust main() detection (ignores comments/strings) ---
    def _strip_strings_and_comments(self, s: str) -> str:
        # Strip string literals
        s = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', s, flags=re.S)
        # Strip // line comments
        s = re.sub(r'//.*$', '', s, flags=re.M)
        # Strip /* ... */ block comments
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
        return s

    def _has_main_function(self, code: str) -> bool:
        clean = self._strip_strings_and_comments(code)
        # Matches: void main(...){...}, main(...){...}, Future<void> main(...){...}, async variants
        pat = re.compile(
            r'^\s*(?:\w+(?:<[^>]*>)?\s+)?'          # optional type e.g., void / Future<void>
            r'main\s*\([^)]*\)\s*(?:async\s*)?\{',  # main(...) [async] {
            re.MULTILINE
        )
        return bool(pat.search(clean))

    def compute_reward(self, solution_code: str, test_code: str):
        # GUARD 1: Empty code check
        if not solution_code.strip() or len(solution_code.strip()) < 10:
            msg = f"Empty/short code violation; Reward={self.empty_code_penalty:.3f}"
            details = {'total': 0, 'passed': 0, 'failed': 0, 'violation': 'empty'}
            return self.empty_code_penalty, msg, details

        # GUARD 2: Main function violation
        if self._has_main_function(solution_code):
            msg = f"Structural violation: candidate defines main(); Reward={self.main_violation_penalty:.3f}"
            details = {'total': 0, 'passed': 0, 'failed': 0, 'violation': 'main'}
            return self.main_violation_penalty, msg, details

        # GUARD 3: Check for obvious nonsense
        nonsense_patterns = [
            r'^[\s\W]*$',  # Only whitespace/special chars
            r'^(sorry|I don\'t|cannot|unable)',  # Refusal patterns
        ]
        clean_code = re.sub(r'\s+', ' ', solution_code.lower().strip())
        for pattern in nonsense_patterns:
            if re.match(pattern, clean_code):
                msg = f"Invalid response pattern; Reward={self.empty_code_penalty:.3f}"
                details = {'total': 0, 'passed': 0, 'failed': 0, 'violation': 'invalid'}
                return self.empty_code_penalty, msg, details

        # NORMAL per-test scoring
        details = self.parse_and_run_individual_tests(solution_code, test_code)
        if details['total'] == 0:
            return 0.0, "No tests found", details

        total = 0.0
        for r in details['individual_results']:
            total += self.base_pass_reward if r['passed'] else self.base_fail_penalty

        if details['passed'] == details['total']:
            total += 2.0  # Perfect bonus

        total = max(-5.0, min(10.0, total))
        msg = (f"Per-test: {details['passed']}/{details['total']} | Reward={total:.3f} "
               f"({self.base_pass_reward}/pass, {self.base_fail_penalty}/fail)")
        return total, msg, details

    # ---------- test parsing (single-line expects only) ----------
    def extract_expect_calls_single_line(self, test_code: str) -> list:
        cases = []
        for ln in test_code.splitlines():
            s = ln.strip()
            if s.startswith('expect(') and 'candidate(' in s:
                cases.append(s if s.endswith(';') else s + ';')
        return cases

    def parse_and_run_individual_tests(self, solution_code: str, test_code: str) -> dict:
        test_cases = self.extract_expect_calls_single_line(test_code)
        results = {'total': len(test_cases), 'passed': 0, 'failed': 0, 'individual_results': []}
        for tc in test_cases:
            ok = self.run_single_expect_test(solution_code, tc, test_code)
            results['individual_results'].append({'test_case': self._short(tc), 'passed': ok})
            results['passed'] += int(ok)
        results['failed'] = results['total'] - results['passed']
        return results

    # ---------- execution ----------
    def run_single_expect_test(self, solution_code: str, test_case: str, full_test_code: str) -> bool:
        try:
            test_imports = self.extract_imports(full_test_code)
            cand_imports = self.extract_imports(solution_code)
            all_imports = self.combine_imports(test_imports, cand_imports)

            helpers = self.extract_helper_functions_safe(full_test_code)
            func_body = self.extract_function_body(solution_code)

            # 1) Preferred: read alias from test file (all your tests do this)
            m = re.search(r"final\s+candidate\s*=\s*(\w+)\s*;", full_test_code)
            if m:
                actual_func_name = m.group(1)
            else:
                # 2) Fallback: infer first function name from solution
                m2 = re.search(r"^[\w<>\[\]\?,\s]+?\s+(\w+)\s*\(", solution_code, re.MULTILINE)
                if not m2:
                    return False
                actual_func_name = m2.group(1)

            program = f"""{all_imports}

{func_body}

{helpers}

void main() {{
  final candidate = {actual_func_name};
  {test_case}
}}
"""
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, 'single_test.dart')
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(program)

                cmd = ['dart', '--disable-dart-dev', 'run', p]
                if self.enable_asserts:
                    cmd.insert(1, '--enable-asserts')

                proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=3)
                return proc.returncode == 0
        except Exception:
            return False

    # ---------- utilities ----------
    def combine_imports(self, a: str, b: str) -> str:
        seen, out = set(), []
        for src in (a, b):
            for ln in src.splitlines():
                key = ln.strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(ln)
        return '\n'.join(out)

    def extract_imports(self, code: str) -> str:
        return '\n'.join([ln for ln in code.splitlines() if ln.strip().startswith(('import ', 'export '))])

    def extract_helper_functions_safe(self, test_code: str) -> str:
        targets = ['expect', 'expectList', 'expectMap']
        lines = test_code.splitlines()
        i, out = 0, []
        while i < len(lines):
            s = lines[i].strip()
            if any(s.startswith(f"void {t}(") for t in targets):
                brace = 0
                j = i
                while j < len(lines):
                    cur = lines[j]
                    out.append(cur)
                    brace += cur.count('{') - cur.count('}')
                    if brace == 0 and cur.strip().endswith('}'):
                        i = j
                        break
                    j += 1
            i += 1
        return '\n'.join(out)

    def extract_function_body(self, code: str) -> str:
        body = []
        for ln in code.splitlines():
            s = ln.strip()
            if not s: continue
            if s.startswith(('import ', 'export ', '@pragma(', 'library ', 'part ', '//', '/*')):
                continue
            body.append(ln)
        return '\n'.join(body)

    def _short(self, s: str, n: int = 100) -> str:
        s = re.sub(r'\s+', ' ', s).strip()
        return s if len(s) <= n else s[:n] + '...'

# Initialize TRUE per-test reward system with STRUCTURAL GUARDS
true_per_test_reward = TruePerTestReward(
    base_pass_reward=1.5,      # 10x increased from 0.15
    base_fail_penalty=-0.5,    # 10x increased from -0.05
    enable_asserts=False,
    main_violation_penalty=-5.0,  # Scaled penalty for main() violations
    empty_code_penalty=-3.0       # Penalty for empty/short code
)

def compute_enhanced_reward(decompiled_code: str, test_code: str) -> tuple:
    """Enhanced reward computation with structural guards and credit assignment"""
    try:
        reward, message, details = true_per_test_reward.compute_reward(decompiled_code, test_code)
        pass_ratio = (details['passed'] / details['total']) if details['total'] else 0.0
        
        # Check for structural violations first
        if 'violation' in details:
            # For violations, we already have the penalty, just return as-is
            return reward, message, pass_ratio
        
        # REWARD SHAPING: Bonus for compilation success (even if tests fail)
        compile_success = reward > -5.0  # Not a catastrophic failure
        if compile_success:
            reward += 0.8  # Significant bonus for compiling code
        
        # REWARD SHAPING: Penalty for empty/short code (redundant but safe)
        if len(decompiled_code.strip()) < 50:
            reward -= 1.0
            
        # REWARD SHAPING: Small bonus for having some structure
        if 'void' in decompiled_code or 'return' in decompiled_code:
            reward += 0.3
            
        return reward, message, pass_ratio
    except Exception as e:
        accelerator.print(f"⚠️ Enhanced reward failed: {e}, using fallback")
        basic_reward, basic_message = run_dart_sandbox(decompiled_code, test_code)
        return basic_reward, f"Fallback: {basic_message}", 0.0

# ============================================================================
# ADAPTIVE TEMPERATURE SCHEDULING
# ============================================================================

def get_adaptive_temp(global_step: int, total_steps: int) -> float:
    """Adaptive temperature scheduling for exploration vs exploitation"""
    base_temp = 1.2  # Higher for exploration
    final_temp = 0.7  # Lower for exploitation
    progress = min(1.0, global_step / total_steps)
    
    # More aggressive exploration in first 30% of training
    if progress < 0.3:
        return base_temp
    # Smooth decay to final temperature
    elif progress < 0.8:
        return base_temp * (1 - (progress - 0.3) / 0.5) + final_temp * ((progress - 0.3) / 0.5)
    else:
        return final_temp

# ============================================================================
# MODEL SETUP
# ============================================================================

# --- PEFT Configuration with DoRA ---
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,
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

# Freeze reference model parameters
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()

model.config.use_cache = False
ref_model.config.use_cache = False
model.config.pretraining_tp = 1
ref_model.config.pretraining_tp = 1

# Enable gradient checkpointing
model.pretrained_model.gradient_checkpointing_enable()

accelerator.print("✓ Models loaded with DoRA")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
accelerator.print(f"✓ DoRA applied: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")

# ============================================================================
# DATASET SETUP
# ============================================================================

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

Include any necessary imports (e.g., import 'dart:math', import 'dart:core') at the beginning. Write ONLY the function implementation - do NOT include test code or main().
Assembly:
{}
Response:
"""

# --- Dataset Loading ---
accelerator.print("\nLoading dataset...")
dataset = load_dataset("json", data_files="grpo_data.jsonl", split="train")
accelerator.print(f"✓ Loaded {len(dataset):,} examples")

def prepare_dataset_entry(example):
    prompt_text = inference_prompt_style.format(
        example['lang'],
        example['lang'],
        example['dart_function_signature'],
        example['assembly']
    )
    return {
        'prompt': prompt_text,
        'tests': example['tests']
    }

dataset = dataset.map(prepare_dataset_entry)

# ============================================================================
# OPTIMIZED TRAINING CONFIGURATION
# ============================================================================

EPOCHS = 3
LEARNING_RATE = 1e-5
TOTAL_STEPS = EPOCHS * (len(dataset) // 2)  # Estimate for adaptive temperature

if SEQUENCE_MODE == "2048":
    PROMPTS_PER_BATCH = 2
    K_SAMPLES = 6
    MAX_LENGTH = 2048
    MAX_NEW_TOKENS = 1536
    GRAD_ACCUM_STEPS = 3
    MINI_BATCH_SIZE = 2
    accelerator.print(f"\n⚙️ Using 2048 token config | Memory: ~95-105 GB")
elif SEQUENCE_MODE == "3072":
    PROMPTS_PER_BATCH = 2
    K_SAMPLES = 4
    MAX_LENGTH = 3072
    MAX_NEW_TOKENS = 2048
    GRAD_ACCUM_STEPS = 4
    MINI_BATCH_SIZE = 2
    accelerator.print(f"\n⚙️ Using 3072 token config (AGGRESSIVE) | Memory: ~115-125 GB")
    accelerator.print(f"   ⚠️ WARNING: Pushing H200 limits!")
else:
    raise ValueError(f"Invalid SEQUENCE_MODE: {SEQUENCE_MODE}")

TOTAL_SEQUENCES = PROMPTS_PER_BATCH * K_SAMPLES

training_arguments = TrainingArguments(
    output_dir="./dart_grpo_dora_optimized_guards",
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

# OPTIMIZED PPO CONFIGURATION
ppo_config = PPOConfig(
    batch_size=TOTAL_SEQUENCES,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    log_with=None,
    ppo_epochs=2,  # Reduced for stability
    early_stopping=False,
    target_kl=0.3,  # Increased for more exploration (was 0.1)
    init_kl_coef=0.1,  # Reduced KL weight (was 0.2)
    adap_kl_ctrl=True,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.2,  # Slightly increased value loss (was 0.1)
    gamma=1.0,
    lam=0.95,
    use_score_scaling=True,  # Important for reward stability
    use_score_norm=True,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# ============================================================================
# OPTIMIZED MAIN TRAINING LOOP WITH STRUCTURAL GUARDS
# ============================================================================

accelerator.print(f"\n{'='*80}")
accelerator.print("STARTING OPTIMIZED DoRA-GRPO TRAINING WITH STRUCTURAL GUARDS")
accelerator.print(f"{'='*80}")
accelerator.print("🎯 OPTIMIZATIONS APPLIED:")
accelerator.print("  • 10x Increased Rewards (1.5/pass, -0.5/fail)")
accelerator.print("  • Relaxed KL Constraints (target_kl=0.3)")
accelerator.print("  • Enhanced Reward Shaping")
accelerator.print("  • Adaptive Temperature Scheduling")
accelerator.print("  • STRUCTURAL GUARDS:")
accelerator.print("    - Main function detection with comment/string stripping")
accelerator.print("    - Empty code penalty (-3.0)")
accelerator.print("    - Nonsense pattern detection")
accelerator.print("  • Advanced Monitoring")
accelerator.print(f"{'='*80}")

# Create distributed sampler
sampler = DistributedSampler(
    dataset, 
    num_replicas=accelerator.num_processes,
    rank=accelerator.process_index, 
    shuffle=True
) if accelerator.num_processes > 1 else None

dataloader = DataLoader(
    dataset,
    batch_size=PROMPTS_PER_BATCH,
    shuffle=(sampler is None),
    sampler=sampler,
    collate_fn=lambda x: {
        'prompts': [item['prompt'] for item in x],
        'tests': [item['tests'] for item in x]
    }
)

# Base generation kwargs (temperature will be set adaptively)
base_gen_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "top_p": 0.9,  # Slightly more focused
    "top_k": 50,   # Added top_k for diversity
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

best_reward = float('-inf')
global_step = 0

# Initialize sample code for logging
sample_code_for_logging = None

for epoch in range(EPOCHS):
    if sampler:
        sampler.set_epoch(epoch)
        
    if accelerator.is_main_process:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    else:
        pbar = dataloader
        
    epoch_rewards = []
    epoch_violations = {'main': 0, 'empty': 0, 'invalid': 0, 'total': 0}
    
    for batch_idx, batch in enumerate(pbar):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize batch pass ratios
        batch_pass_ratios = []
        
        # === 1. Set Adaptive Temperature ===
        current_temp = get_adaptive_temp(global_step, TOTAL_STEPS)
        gen_kwargs = {**base_gen_kwargs, "temperature": current_temp}
        
        # === 2. Tokenize Prompts ===
        all_query_tensors_unpadded = []
        all_query_tensors_padded = []
        all_query_masks_padded = []
        all_prompt_lengths = []
        
        for prompt in batch['prompts']:
            tokens_unpadded = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(accelerator.device)
            all_query_tensors_unpadded.append(tokens_unpadded.input_ids.squeeze(0))
            prompt_len = tokens_unpadded.input_ids.shape[1]
            all_prompt_lengths.append(prompt_len)
            
            tokens_padded = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(accelerator.device)
            all_query_tensors_padded.append(tokens_padded.input_ids.squeeze(0))
            all_query_masks_padded.append(tokens_padded.attention_mask.squeeze(0))
        
        # === 3. Generate K samples per prompt ===
        all_response_tensors = []
        all_response_masks = []
        raw_reward_groups = []
        
        ppo_trainer.model.eval()
        for idx, (q_tensor_pad, q_mask_pad, prompt_len) in enumerate(zip(
            all_query_tensors_padded, all_query_masks_padded, all_prompt_lengths
        )):
            q_batch = q_tensor_pad.unsqueeze(0)
            mask_batch = q_mask_pad.unsqueeze(0)
            group_rewards = []
            
            with torch.no_grad():
                generated = accelerator.unwrap_model(ppo_trainer.model).generate(
                    q_batch,
                    attention_mask=mask_batch,
                    num_return_sequences=K_SAMPLES,
                    **gen_kwargs
                )
            
            input_len = q_batch.size(1)
            
            for seq in generated:
                resp_tokens = seq[input_len:]
                eos_pos = (resp_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    resp_tokens = resp_tokens[:eos_pos[0] + 1]
                
                assert resp_tokens.numel() > 0, "Empty response after generation slice"
                
                all_response_tensors.append(resp_tokens)
                all_response_masks.append(
                    torch.ones_like(resp_tokens, dtype=torch.long, device=resp_tokens.device)
                )
                
                text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
                tests = batch['tests'][idx]
                
                # Extract code using robust regex
                m = re.search(r"```(?:dart)?\s*([\s\S]*?)```", text, re.IGNORECASE)
                code = (m.group(1) if m else text).strip()
                
                # Store sample code for logging (only from first prompt in batch)
                if sample_code_for_logging is None and idx == 0:
                    sample_code_for_logging = code
                
                # USE ENHANCED REWARD SYSTEM WITH STRUCTURAL GUARDS
                reward, message, pass_ratio = compute_enhanced_reward(code, tests)
                group_rewards.append(float(reward))
                batch_pass_ratios.append(float(pass_ratio))
                
                # Track violations for monitoring
                if "violation" in message:
                    epoch_violations['total'] += 1
                    if "main" in message:
                        epoch_violations['main'] += 1
                    elif "empty" in message:
                        epoch_violations['empty'] += 1
                    elif "invalid" in message:
                        epoch_violations['invalid'] += 1
            
            raw_reward_groups.append(group_rewards)
            
            del generated
            torch.cuda.empty_cache()
        
        ppo_trainer.model.train()
        
        # === 4. GRPO: Group-relative advantages ===
        all_shaped_scores = []
        all_raw_rewards_logging = []
        
        for group_rewards in raw_reward_groups:
            if len(group_rewards) == 0:
                continue
                
            r = np.array(group_rewards, dtype=np.float32)
            r_mean = float(r.mean())
            adv = r - r_mean
            
            if len(r) > 1:
                ranks = rankdata(-r, method='average')
                util = (len(r) - ranks) / (len(r) - 1)
                util = util - util.mean()
            else:
                util = np.zeros_like(r)
            
            shaped = adv + (GRPO_BONUS_WEIGHT * util)
            
            for s, rr in zip(shaped, r):
                if np.isnan(s) or np.isinf(s):
                    s = 0.0
                all_shaped_scores.append(
                    torch.tensor(float(s), dtype=torch.float32, device=accelerator.device)
                )
                all_raw_rewards_logging.append(float(rr))
        
        # === 5. Repeat UNPADDED Queries for PPO ===
        queries_repeated = []
        for i, query in enumerate(all_query_tensors_unpadded):
            group_size = len(raw_reward_groups[i]) if i < len(raw_reward_groups) else K_SAMPLES
            queries_repeated.extend([query] * group_size)
        
        # === 6. Validate tensor alignment ===
        assert len(queries_repeated) == len(all_response_tensors) == len(all_shaped_scores) == len(all_response_masks), \
            f"Tensor length mismatch: Q{len(queries_repeated)} R{len(all_response_tensors)} S{len(all_shaped_scores)} M{len(all_response_masks)}"
        
        if len(queries_repeated) == 0:
            accelerator.print("⚠️ Skipping empty batch")
            continue
        
        # === 7. PPO Step ===
        try:
            stats = ppo_trainer.step(
                queries=queries_repeated,
                responses=all_response_tensors,
                scores=all_shaped_scores,
                response_masks=all_response_masks
            )
        except Exception as e:
            accelerator.print(f"❌ PPO step failed: {e}")
            continue
        
        # === 8. ENHANCED MONITORING WITH VIOLATION TRACKING ===
        mean_raw_reward = sum(all_raw_rewards_logging) / len(all_raw_rewards_logging) if all_raw_rewards_logging else 0.0
        mean_shaped_score = sum(s.item() for s in all_shaped_scores) / len(all_shaped_scores) if all_shaped_scores else 0.0
        
        # Use pass ratio instead of binary success rate
        success_rate = float(np.mean(batch_pass_ratios)) if batch_pass_ratios else 0.0
        
        epoch_rewards.extend(all_raw_rewards_logging)
        
        if accelerator.is_main_process:
            allocated = torch.cuda.memory_allocated() / 1e9
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            
            pbar.set_description(
                f"E{epoch+1} S{global_step} | "
                f"R_Raw:{mean_raw_reward:.3f} | "
                f"R_Shape:{mean_shaped_score:.3f} | "
                f"Pass:{success_rate*100:.1f}% | "
                f"Temp:{current_temp:.2f} | "
                f"Mem:{allocated:.0f}GB"
            )
            
            # ADVANCED MONITORING every 20 steps
            if global_step % 20 == 0 and all_raw_rewards_logging:
                rewards_np = np.array(all_raw_rewards_logging)
                accelerator.print(
                    f"📊 OPTIMIZED Stats: "
                    f"min={rewards_np.min():.3f}, max={rewards_np.max():.3f}, "
                    f"mean={rewards_np.mean():.3f}, std={rewards_np.std():.3f}"
                )
                
                # Violation tracking
                if epoch_violations['total'] > 0:
                    accelerator.print(
                        f"🚫 Violations: {epoch_violations['total']} total "
                        f"(main: {epoch_violations['main']}, "
                        f"empty: {epoch_violations['empty']}, "
                        f"invalid: {epoch_violations['invalid']})"
                    )
                
                # KL divergence monitoring
                if 'objective/kl' in stats:
                    kl_value = stats['objective/kl']
                    accelerator.print(f"📈 KL Divergence: {kl_value:.4f} (target: {ppo_config.target_kl})")
                
                # Policy loss vs value loss
                if 'policy/policy_loss' in stats and 'val/value_loss' in stats:
                    policy_loss = stats['policy/policy_loss']
                    value_loss = stats['val/value_loss']
                    accelerator.print(f"📉 Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
                
                # Clip fraction monitoring
                if 'policy/clipfrac' in stats:
                    clipfrac = stats['policy/clipfrac']
                    accelerator.print(f"✂️ Clip Fraction: {clipfrac:.4f}")
                
                # Log individual test performance using sample code
                if sample_code_for_logging and raw_reward_groups and len(raw_reward_groups[0]) > 0:
                    td = true_per_test_reward.parse_and_run_individual_tests(sample_code_for_logging, batch['tests'][0])
                    accelerator.print(
                        f"🧪 Test Performance: {td['passed']}/{td['total']} passed "
                        f"({(td['passed']/td['total']*100 if td['total'] else 0):.1f}%)"
                    )
                    
                    # Reset sample code for next logging cycle
                    sample_code_for_logging = None
        
        if global_step % 10 == 0:
            torch.cuda.reset_peak_memory_stats()
        
        global_step += 1
        
        # === 9. Checkpointing ===
        if batch_idx % 50 == 0 and batch_idx > 0 and accelerator.is_main_process:
            ckpt_dir = f"{training_arguments.output_dir}/ckpt-e{epoch+1}-s{global_step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            accelerator.print(f"✓ Checkpoint saved: {ckpt_dir}")
        
        # === 10. Cleanup ===
        del (
            all_query_tensors_unpadded, all_query_tensors_padded, all_query_masks_padded,
            all_response_tensors, all_response_masks, all_shaped_scores, queries_repeated
        )
        torch.cuda.empty_cache()
        gc.collect()
    
    # === Epoch Summary ===
    if accelerator.is_main_process:
        epoch_mean = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        epoch_success = sum(1 for r in epoch_rewards if r >= 1.0) / len(epoch_rewards) if epoch_rewards else 0.0
        
        accelerator.print(f"\n{'='*80}")
        accelerator.print(f"EPOCH {epoch+1} SUMMARY")
        accelerator.print(f"  Mean Raw Reward: {epoch_mean:.4f}")
        accelerator.print(f"  Success Rate: {epoch_success*100:.2f}%")
        accelerator.print(f"  Violations: {epoch_violations['total']} "
                         f"(main: {epoch_violations['main']}, "
                         f"empty: {epoch_violations['empty']}, "
                         f"invalid: {epoch_violations['invalid']})")
        accelerator.print(f"  Total Steps: {global_step}")
        accelerator.print(f"  Current Temperature: {current_temp:.2f}")
        accelerator.print(f"{'='*80}")
        
        if epoch_mean > best_reward:
            best_reward = epoch_mean
            best_dir = f"{training_arguments.output_dir}/best"
            os.makedirs(best_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            accelerator.print(f"🏆 New best model saved: {best_dir} (Reward: {best_reward:.4f})")

# === Final Save ===
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    final_dir = training_arguments.output_dir
    accelerator.unwrap_model(model).save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    accelerator.print(f"\n{'='*80}")
    accelerator.print(f"✓ FINAL MODEL SAVED: {final_dir}")
    accelerator.print(f"✓ BEST MODEL SAVED: {training_arguments.output_dir}/best")
    accelerator.print(f"  Best Raw Reward: {best_reward:.4f}")
    accelerator.print(f"  Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    accelerator.print(f"  Total Training Steps: {global_step}")
    accelerator.print("\n" + "="*80)
    accelerator.print("🎉 OPTIMIZED TRUE PER-TEST DoRA-GRPO TRAINING WITH STRUCTURAL GUARDS COMPLETED!")
    accelerator.print("="*80)
