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
from tqdm.auto import tqdm
from accelerate import Accelerator
import gc
from scipy.stats import rankdata

# ============================================================================
# H200 DoRA-OPTIMIZED GRPO TRAINING SCRIPT - FULLY FIXED VERSION
# Hardware: NVIDIA H200 (141GB HBM3e)
# Features: DoRA enabled + Long sequences (2048 or 3072)
# Strategy: Smaller batches, aggressive memory management
# FIXES: Multi-extraction, KL stability, reward normalization, gradient clipping
# ============================================================================

print("🚀 H200 DoRA-Optimized GRPO Training Script (FULLY FIXED)")
print("=" * 80)

# CONFIGURATION SELECTOR
# ========================================================================
# Choose your configuration here:
SEQUENCE_MODE = "2048"  # Options: "2048" or "3072"
# ========================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

accelerator = Accelerator(mixed_precision="bf16")

model_dir = "raafatabualazm/decompiler-v1"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

accelerator.print("=" * 80)
accelerator.print("CONFIGURATION (FULLY FIXED)")
accelerator.print("=" * 80)
accelerator.print(f"  Device: {accelerator.device}")
accelerator.print(f"  Mixed Precision: bf16")
accelerator.print(f"  Available Memory: ~141GB (H200)")
accelerator.print(f"  Sequence Mode: {SEQUENCE_MODE} tokens")
accelerator.print(f"  DoRA: ENABLED")
accelerator.print(f"  Memory Optimization: AGGRESSIVE")
accelerator.print(f"  🔧 FIXES APPLIED:")
accelerator.print(f"     • Multi-extraction (tests all code snippets)")
accelerator.print(f"     • KL stability (init_coef=0.3, target=0.1)")
accelerator.print(f"     • Reward normalization")
accelerator.print(f"     • Gradient clipping (max_norm=1.0)")
accelerator.print(f"     • Reference model on GPU")
accelerator.print(f"     • Enhanced debug logging")
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
                ['dart', '--disable-dart-dev', 'run', test_filepath],
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
            # Match any return type and optional generics before the name.
            if any(re.match(rf'^\s*(?:[\w<>\[\]\?,\s]+?\s+)?{t}\s*\(', s) for t in targets):
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
    base_pass_reward=1.5,
    base_fail_penalty=-0.5,
    enable_asserts=False,
    main_violation_penalty=-5.0,
    empty_code_penalty=-3.0
)

def compute_enhanced_reward(decompiled_code: str, test_code: str) -> tuple:
    """Enhanced reward computation with structural guards and credit assignment"""
    try:
        reward, message, details = true_per_test_reward.compute_reward(decompiled_code, test_code)
        pass_ratio = (details['passed'] / details['total']) if details['total'] else 0.0
        
        # Check for structural violations first
        if 'violation' in details:
            return reward, message, pass_ratio
        
        # REWARD SHAPING
        compile_success = reward > -5.0  # Not a catastrophic failure
        if compile_success:
            reward += 0.8  # Significant bonus for compiling code
        if len(decompiled_code.strip()) < 50:
            reward -= 1.0
        if 'void' in decompiled_code or 'return' in decompiled_code:
            reward += 0.3
            
        return reward, message, pass_ratio
    except Exception as e:
        accelerator.print(f"⚠️ Enhanced reward failed: {e}, using fallback")
        basic_reward, basic_message = run_dart_sandbox(decompiled_code, test_code)
        return basic_reward, f"Fallback: {basic_message}", 0.0

# ============================================================================
# MULTI-EXTRACTION: Find ALL code snippets and test each one
# ============================================================================

def extract_all_dart_code_snippets(completion: str) -> list:
    """
    Extract ALL potential Dart code snippets from completion.
    Returns: List of (code, extraction_method) tuples
    """
    snippets = []
    
    # Pattern 1: All ```dart ... ``` blocks
    for match in re.finditer(r"```dart\s*(.*?)\s*```", completion, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if code:
            snippets.append((code, "dart_fenced"))
    
    # Pattern 2: All ``` ... ``` blocks (no language specified)
    for match in re.finditer(r"```\s*(.*?)\s*```", completion, re.DOTALL):
        code = match.group(1).strip()
        # Verify it looks like Dart
        if code and re.search(r"(List<|String|int|double|bool|void)\s+\w+\s*\(", code):
            # Don't add if we already got this exact code from dart_fenced
            if not any(snippet[0] == code and snippet[1] == "dart_fenced" for snippet in snippets):
                snippets.append((code, "generic_fenced"))
    
    # Pattern 3: Find all function signatures and extract surrounding code
    for match in re.finditer(r"(List<String>|String|int|double|bool|void)\s+\w+\s*\([^)]*\)\s*\{", completion, re.DOTALL):
        start_idx = match.start()
        
        # Extract from function start
        code_part = completion[start_idx:]
        
        # Try to find the end of the function (balanced braces)
        brace_count = 0
        in_function = False
        end_idx = 0
        
        for i, char in enumerate(code_part):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            code = code_part[:end_idx].strip()
        else:
            # Fallback: take until next section marker
            end_match = re.search(r"\n\n(###|---|\*\*|Explanation|Note:)", code_part)
            code = code_part[:end_match.start()].strip() if end_match else code_part.strip()
        
        # Don't add if already extracted
        if code and not any(snippet[0] == code for snippet in snippets):
            snippets.append((code, "signature_search"))
    
    # Pattern 4: Extract code between common delimiters (even without ```)
    for pattern in [r"(?:^|\n)([A-Z][\w<>,\s]*\s+\w+\s*\([^)]*\)\s*\{[^}]*\})", 
                    r"(?:Solution:|Answer:|Code:|Response:)\s*\n(.*?)(?:\n\n|\Z)"]:
        for match in re.finditer(pattern, completion, re.DOTALL | re.MULTILINE):
            code = match.group(1).strip()
            if code and len(code) > 30 and not any(snippet[0] == code for snippet in snippets):
                if re.search(r"(return|if|for|while|List|String|void|int|double|bool)", code):
                    snippets.append((code, "delimiter_search"))
    
    return snippets


def compute_best_reward_from_all_snippets(completion: str, test_code: str, prompt: str,
                                         current_step: int = 0, 
                                         should_log: bool = False) -> tuple:
    """
    Extract all code snippets, test each, and return the best reward.
    Returns: (best_reward, best_message, best_pass_ratio, best_code, extraction_method, num_attempts)
    """
    all_snippets = extract_all_dart_code_snippets(completion)
    
    # 🔍 DEBUG: Log extraction attempts
    if should_log and len(all_snippets) > 1:
        accelerator.print(f"   🔎 Found {len(all_snippets)} code snippets to test")
    
    if not all_snippets:
        # Fallback: treat entire completion as raw code if it has code-like content
        if len(completion.strip()) > 50 and re.search(r"(return|if|for|while|List|String)", completion):
            all_snippets = [(completion.strip(), "raw_text")]
        else:
            return -2.0, "No code snippets found", 0.0, None, "none", 0
    
    best_reward = float('-inf')
    best_message = ""
    best_pass_ratio = 0.0
    best_code = None
    best_method = "none"
    
    # Test each snippet and keep the best
    for idx, (code_snippet, method) in enumerate(all_snippets):
        try:
            reward, message, pass_ratio = compute_enhanced_reward(code_snippet, test_code)
            
            # 🔍 DEBUG: Log each attempt (only for first few batches)
            if should_log and (idx == 0 or reward > best_reward):
                if current_step < 20:  # Only log early in training
                    accelerator.print(f"      Snippet {idx+1}/{len(all_snippets)} ({method}): reward={reward:.3f}, pass_ratio={pass_ratio:.2f}")
            
            if reward > best_reward:
                best_reward = reward
                best_message = message
                best_pass_ratio = pass_ratio
                best_code = code_snippet
                best_method = method
                
                # Early exit if we found a perfect solution
                if reward >= 9.0:  # Near-perfect score
                    break
                    
        except Exception as e:
            # Skip problematic snippets
            continue
    
    # If all snippets failed, return the best (least bad) one
    if best_code is None and all_snippets:
        best_code = all_snippets[0][0]
        best_method = all_snippets[0][1]
        best_reward = -1.0
        best_message = "All extractions failed testing"
    
    return best_reward, best_message, best_pass_ratio, best_code, best_method, len(all_snippets)

# --- PEFT Configuration with DoRA ---
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=32,
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

# 🔧 FIX: Move reference model to GPU for numerical stability
ref_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},  # Changed from "cpu"
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
accelerator.print("✓ Reference model on GPU for stability")

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

# --- Dataset Loading ---
accelerator.print("\nLoading dataset...")
dataset = load_dataset("json", data_files="grpo_data.jsonl", split="train")
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
    accelerator.print("\n⚙️  Using 2048 token configuration")
    PROMPTS_PER_BATCH = 1
    K_SAMPLES = 1
    MAX_LENGTH = 6144
    MAX_NEW_TOKENS = 4096
    GRAD_ACCUM_STEPS = 2
    MINI_BATCH_SIZE = 1
    accelerator.print(f"   • Prompts/batch: {PROMPTS_PER_BATCH}")
    accelerator.print(f"   • Samples/prompt: {K_SAMPLES}")
    accelerator.print(f"   • Total sequences: {PROMPTS_PER_BATCH * K_SAMPLES}")
    accelerator.print(f"   • Expected memory: 95-105 GB")
    
elif SEQUENCE_MODE == "3072":
    accelerator.print("\n⚙️  Using 3072 token configuration (AGGRESSIVE)")
    PROMPTS_PER_BATCH = 2
    K_SAMPLES = 2
    MAX_LENGTH = 8192
    MAX_NEW_TOKENS = 8192
    GRAD_ACCUM_STEPS = 4
    MINI_BATCH_SIZE = 2
    accelerator.print(f"   • Prompts/batch: {PROMPTS_PER_BATCH}")
    accelerator.print(f"   • Samples/prompt: {K_SAMPLES}")
    accelerator.print(f"   • Total sequences: {PROMPTS_PER_BATCH * K_SAMPLES}")
    accelerator.print(f"   • Expected memory: 115-125 GB")
    accelerator.print(f"   • ⚠️  WARNING: Pushing memory limits!")
else:
    raise ValueError(f"Invalid SEQUENCE_MODE: {SEQUENCE_MODE}")

accelerator.print("=" * 80)

# 🔧 FIX: Add gradient clipping
training_arguments = TrainingArguments(
    output_dir="./dart_grpo_dora_h200_fixed",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PROMPTS_PER_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    max_grad_norm=1.0,  # Added gradient clipping
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to="none"
)

# --- PPO Configuration (FIXED FOR KL STABILITY) ---
TOTAL_SEQUENCES = PROMPTS_PER_BATCH * K_SAMPLES

# 🔧 FIX: Proper KL configuration for stability
ppo_config = PPOConfig(
    batch_size=TOTAL_SEQUENCES,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    log_with=None,
    ppo_epochs=3,
    early_stopping=False,
    target_kl=0.1,        # Reduced back to 0.1 for stability
    init_kl_coef=0.3,     # Increased from 0.1 to 0.3 for better control
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
accelerator.print("STARTING DoRA TRAINING WITH LONG SEQUENCES (FULLY FIXED)")
accelerator.print(f"{'='*80}")
accelerator.print(f"Total Epochs: {EPOCHS}")
accelerator.print(f"Sequence length: {MAX_LENGTH}")
accelerator.print(f"Max new tokens: {MAX_NEW_TOKENS}")
accelerator.print(f"Batch: {PROMPTS_PER_BATCH} prompts × {K_SAMPLES} samples = {TOTAL_SEQUENCES} sequences")
accelerator.print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} (effective: {PROMPTS_PER_BATCH * GRAD_ACCUM_STEPS} prompts)")
accelerator.print(f"KL Config: target={ppo_config.target_kl}, init_coef={ppo_config.init_kl_coef}")
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
    "temperature": 0.7,       # Reduced from 0.65 for stability
    "top_p": 0.9,             # Reduced from 0.95
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

best_reward = float('-inf')
global_step = 0

# 🔧 FIX: Added delimiter_search to extraction stats
extraction_stats = {
    'dart_fenced': 0, 
    'generic_fenced': 0, 
    'signature_search': 0,
    'delimiter_search': 0,  # Added
    'raw_text': 0, 
    'none': 0
}

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
            
            # 🔧 DEBUG: Print first generation
            if batch_idx == 0 and query_idx == 0 and accelerator.is_main_process:
                accelerator.print(f"\n🔍 FIRST GENERATION (Full):")
                accelerator.print("=" * 80)
                accelerator.print(decoded_texts[0][:1000])
                accelerator.print("=" * 80 + "\n")
            
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
        extraction_attempts_total = 0

        example_idx = 0
        for ex_prompt, ex_tests in zip(batch['prompts'], batch['tests']):
            group_rewards = []
            
            for i in range(K_SAMPLES):
                gen_idx = example_idx * K_SAMPLES + i
                gen_text = all_generated_texts[gen_idx]
                completion = gen_text[len(ex_prompt):]
                
                # 🔧 DEBUG: Print sample completions every 10 batches
                if batch_idx % 10 == 0 and i == 0 and example_idx == 0 and accelerator.is_main_process:
                    accelerator.print(f"\n📝 Sample completion (batch {batch_idx}):")
                    accelerator.print("=" * 80)
                    accelerator.print(completion[:500])
                    accelerator.print("=" * 80 + "\n")
                
                # 🔧 MULTI-EXTRACTION: Test all code snippets and use best reward
                should_log = (batch_idx < 10 and accelerator.is_main_process)
                reward_val, log_msg, pass_ratio, best_code, extraction_method, num_attempts = \
                    compute_best_reward_from_all_snippets(
                        completion, 
                        ex_tests, 
                        ex_prompt,
                        current_step=global_step,
                        should_log=should_log
                    )
                
                extraction_attempts_total += num_attempts
                extraction_stats[extraction_method] += 1
                
                if reward_val >= 1.0:
                    successful_completions += 1
                
                # 🔍 Log successful extractions
                if num_attempts > 1 and should_log:
                    accelerator.print(f"   ✓ Tested {num_attempts} snippets, best: {extraction_method} (reward={reward_val:.3f})")

                group_rewards.append(reward_val)
                raw_rewards_log.append(reward_val)
            
            # GRPO: Calculate advantage (group-wise)
            group_mean = sum(group_rewards) / len(group_rewards)
            group_advantages = [
                torch.tensor(r - group_mean, dtype=torch.float32).to(accelerator.device) 
                for r in group_rewards
            ]
            rewards.extend(group_advantages)
            
            example_idx += 1

        # 🔧 FIX: Normalize rewards across all groups
        if len(rewards) > 1:
            rewards_tensor = torch.stack(rewards)
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std() + 1e-8
            rewards = [(r - rewards_mean) / rewards_std for r in rewards]

        # 🔧 Log average extraction attempts
        avg_attempts = extraction_attempts_total / len(raw_rewards_log) if raw_rewards_log else 0
        if accelerator.is_main_process and batch_idx % 10 == 0:
            accelerator.print(f"   Avg snippets per generation: {avg_attempts:.1f}")
                
        # --- PPO Optimization Step ---
        stats = ppo_trainer.step(queries_repeated, response_tensors, rewards)
        
        # 🔧 ENHANCED KL MONITORING
        if 'objective/kl' in stats:
            kl_value = stats['objective/kl']
            if kl_value < -10.0 and accelerator.is_main_process:
                accelerator.print(f"\n⚠️  WARNING: KL divergence negative: {kl_value:.2f}")
                accelerator.print("    Monitoring for instability...")
            if kl_value < -100.0 and accelerator.is_main_process:
                accelerator.print(f"\n❌ CRITICAL: KL collapse detected ({kl_value:.2f})")
                accelerator.print("    Consider stopping and adjusting hyperparameters\n")
        
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
            
            # Get KL value for display
            kl_display = stats.get('objective/kl', 0.0) if stats else 0.0
            
            pbar.set_description(
                f"E{epoch+1} | S{global_step} | "
                f"R:{mean_raw_reward:.3f} | "
                f"Acc:{success_rate*100:.1f}% | "
                f"KL:{kl_display:.2f} | "
                f"Mem:{allocated:.0f}GB"
            )
            
            # 🔧 ENHANCED MONITORING every 20 steps
            if global_step % 20 == 0:
                accelerator.print(f"\n📊 Step {global_step} Statistics:")
                accelerator.print(f"   Rewards: min={min(raw_rewards_log):.3f}, max={max(raw_rewards_log):.3f}, mean={mean_raw_reward:.3f}")
                accelerator.print(f"   Success Rate: {success_rate*100:.1f}%")
                accelerator.print(f"   Extraction Methods: {extraction_stats}")
                accelerator.print(f"   Avg Snippets/Gen: {avg_attempts:.2f}")
                if 'objective/kl' in stats:
                    accelerator.print(f"   KL Divergence: {stats['objective/kl']:.4f} (target: {ppo_config.target_kl})")
                if 'ppo/policy/approxkl' in stats:
                    accelerator.print(f"   Approx KL: {stats['ppo/policy/approxkl']:.4f}")
                if 'policy/clipfrac' in stats:
                    accelerator.print(f"   Clip Fraction: {stats['policy/clipfrac']:.4f}")
                accelerator.print("")
            
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
                f.write(f"Extraction Stats: {extraction_stats}\n")
                f.write(f"Avg Snippets/Gen: {avg_attempts:.2f}\n")
            
            accelerator.print(f"\n💾 Checkpoint saved: {checkpoint_dir}")
    
    # End of epoch
    if accelerator.is_main_process:
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        epoch_success_rate = sum(1 for r in epoch_rewards if r >= 1.0) / len(epoch_rewards) if epoch_rewards else 0.0
        
        accelerator.print(f"\n{'='*80}")
        accelerator.print(f"EPOCH {epoch+1} SUMMARY")
        accelerator.print(f"{'='*80}")
        accelerator.print(f"Mean Reward:   {epoch_mean_reward:.4f}")
        accelerator.print(f"Success Rate:   {epoch_success_rate*100:.2f}%")
        accelerator.print(f"Total Samples:  {len(epoch_rewards):,}")
        accelerator.print(f"Peak Memory:    {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        accelerator.print(f"Extraction Stats: {extraction_stats}")
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
    accelerator.print(f"  Final extraction stats: {extraction_stats}")

accelerator.print("\n" + "="*80)
accelerator.print("🎉 DoRA TRAINING COMPLETED (FULLY FIXED)!")
accelerator.print("="*80)