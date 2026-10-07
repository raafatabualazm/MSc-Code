"""
Causal GRPO (Group Relative Policy Optimization) training script with unit-test sandbox rewards (Antigravity version).
Optimized for high performance and low memory using PEFT adapter-disabling context to calculate reference model logits.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
except Exception as e:
    pass

import os
import re
import json
import argparse
import tempfile
import shutil
import subprocess
import gc
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
                encoding='utf-8',
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
                 base_reward: float = -1.0,
                 pass_ratio_reward: float = 8.0,
                 perfect_bonus: float = 2.0,
                 enable_asserts: bool = False,
                 main_violation_penalty: float = -5.0,
                 empty_code_penalty: float = -3.0):
        self.base_reward = base_reward
        self.pass_ratio_reward = pass_ratio_reward
        self.perfect_bonus = perfect_bonus
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

    def _extract_code_candidate(self, code: str) -> str:
        m = re.search(r"```(?:dart)?\s*(.*?)```", code, flags=re.S | re.I)
        if m:
            return m.group(1).strip()
        return code.strip()

    def _remove_main_function(self, code: str) -> str:
        code = self._extract_code_candidate(code)
        while True:
            main_match = re.search(
                r"^\s*(?:\w+(?:<[^>]*>)?\s+)?main\s*\([^)]*\)\s*(?:async\s*)?\{",
                code,
                flags=re.MULTILINE,
            )
            if not main_match:
                return code.strip()

            start = main_match.start()
            depth = 0
            i = main_match.end() - 1
            end = None
            while i < len(code):
                ch = code[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                i += 1

            if end is None:
                return code[:start].strip()
            code = (code[:start] + code[end:]).strip()

    def _prepare_solution_for_tests(self, code: str) -> str:
        code = self._remove_main_function(code)
        code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    def compute_reward(self, solution_code: str, test_code: str):
        solution_code = self._prepare_solution_for_tests(solution_code)
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

        pass_ratio = details['passed'] / details['total']
        total = self.base_reward + self.pass_ratio_reward * pass_ratio

        if details['passed'] == details['total']:
            total += self.perfect_bonus

        total = max(-5.0, min(10.0, total))
        msg = (f"Per-test: {details['passed']}/{details['total']} | Reward={total:.3f} "
               f"(pass_ratio={pass_ratio:.3f})")
        return total, msg, details

    def extract_expect_calls_single_line(self, test_code: str) -> list:
        cases = []
        for ln in test_code.splitlines():
            s = ln.strip()
            if s.startswith('expect(') and 'candidate(' in s:
                cases.append(s if s.endswith(';') else s + ';')
        return cases

    def parse_and_run_individual_tests(self, solution_code: str, test_code: str) -> dict:
        solution_code = self._prepare_solution_for_tests(solution_code)
        test_cases = self.extract_expect_calls_single_line(test_code)
        results = {'total': len(test_cases), 'passed': 0, 'failed': 0, 'individual_results': []}
        for tc in test_cases:
            ok = self.run_single_expect_test(solution_code, tc, test_code)
            results['individual_results'].append({'test_case': self._short(tc), 'passed': ok})
            results['passed'] += int(ok)
        results['failed'] = results['total'] - results['passed']
        return results

    def run_single_expect_test(self, solution_code: str, test_case: str, full_test_code: str) -> bool:
        try:
            test_imports = self.extract_imports(full_test_code)
            cand_imports = self.extract_imports(solution_code)
            all_imports = self.combine_imports(test_imports, cand_imports)

            helpers = self.extract_helper_functions_safe(full_test_code)
            func_body = self.extract_function_body(solution_code)

            m = re.search(r"final\s+candidate\s*=\s*(\w+)\s*;", full_test_code)
            if m:
                actual_func_name = m.group(1)
            else:
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
            is_def = (
                any(re.match(rf'^\s*[\w<>\[\]\?,\s]+?\s+{t}\s*\(', s) for t in targets)
                and not s.endswith(';')
                and '{' in s
            )
            if is_def:
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
        code = self._prepare_solution_for_tests(code)
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

# Initialize true per-test reward system
true_per_test_reward = TruePerTestReward(
    base_reward=-1.0,
    pass_ratio_reward=8.0,
    perfect_bonus=2.0,
    enable_asserts=False,
    main_violation_penalty=-5.0,
    empty_code_penalty=-3.0
)

def compute_enhanced_reward(decompiled_code: str, test_code: str) -> tuple:
    """Enhanced reward computation with structural guards and credit assignment"""
    try:
        reward, message, details = true_per_test_reward.compute_reward(decompiled_code, test_code)
        pass_ratio = (details['passed'] / details['total']) if details['total'] else 0.0
        
        if 'violation' in details:
            return reward, message, pass_ratio
        
        # Keep the reward dominated by functional behavior. Tiny shaping only
        # helps break ties early; it should never overpower test results.
        if pass_ratio == 0.0 and len(decompiled_code.strip()) < 50:
            reward -= 0.5
        if re.search(r"\b(?:List<[^>]+>|String|int|double|bool|void)\s+\w+\s*\(", decompiled_code):
            reward += 0.2
        reward = max(-5.0, min(10.0, reward))
            
        return reward, message, pass_ratio
    except Exception as e:
        basic_reward, basic_message = run_dart_sandbox(decompiled_code, test_code)
        return basic_reward, f"Fallback: {basic_message}", 0.0

# ============================================================================
# MULTI-EXTRACTION Snipets parsing
# ============================================================================

def extract_all_dart_code_snippets(completion: str) -> list:
    """Extract ALL potential Dart code snippets from completion."""
    snippets = []
    
    # Pattern 1: Fenced blocks
    for match in re.finditer(r"```dart\s*(.*?)\s*```", completion, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if code:
            snippets.append((code, "dart_fenced"))
    
    for match in re.finditer(r"```\s*(.*?)\s*```", completion, re.DOTALL):
        code = match.group(1).strip()
        if code and re.search(r"(List<|String|int|double|bool|void)\s+\w+\s*\(", code):
            if not any(snippet[0] == code and snippet[1] == "dart_fenced" for snippet in snippets):
                snippets.append((code, "generic_fenced"))
    
    # Pattern 2: Signature based
    for match in re.finditer(r"(List<String>|String|int|double|bool|void)\s+\w+\s*\([^)]*\)\s*\{", completion, re.DOTALL):
        start_idx = match.start()
        code_part = completion[start_idx:]
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
            end_match = re.search(r"\n\n(###|---|\*\*|Explanation|Note:)", code_part)
            code = code_part[:end_match.start()].strip() if end_match else code_part.strip()
        if code and not any(snippet[0] == code for snippet in snippets):
            snippets.append((code, "signature_search"))
            
    # Pattern 3: Delimiter blocks
    for pattern in [r"(?:^|\n)([A-Z][\w<>,\s]*\s+\w+\s*\([^)]*\)\s*\{[^}]*\})", 
                    r"(?:Solution:|Answer:|Code:|Response:)\s*\n(.*?)(?:\n\n|\Z)"]:
        for match in re.finditer(pattern, completion, re.DOTALL | re.MULTILINE):
            code = match.group(1).strip()
            if code and len(code) > 30 and not any(snippet[0] == code for snippet in snippets):
                if re.search(r"(return|if|for|while|List|String|void|int|double|bool)", code):
                    snippets.append((code, "delimiter_search"))
    
    return snippets

def compute_best_reward_from_all_snippets(completion: str, test_code: str) -> tuple:
    all_snippets = extract_all_dart_code_snippets(completion)
    if not all_snippets:
        if len(completion.strip()) > 50 and re.search(r"(return|if|for|while|List|String)", completion):
            all_snippets = [(completion.strip(), "raw_text")]
        else:
            return -2.0, "No code snippets found", 0.0, None, "none"
            
    best_reward = float('-inf')
    best_message = ""
    best_pass_ratio = 0.0
    best_code = None
    best_method = "none"
    
    for idx, (code_snippet, method) in enumerate(all_snippets):
        try:
            reward, message, pass_ratio = compute_enhanced_reward(code_snippet, test_code)
            if reward > best_reward:
                best_reward = reward
                best_message = message
                best_pass_ratio = pass_ratio
                best_code = code_snippet
                best_method = method
                if reward >= 9.0:
                    break
        except Exception:
            continue
            
    if best_code is None and all_snippets:
        best_code = all_snippets[0][0]
        best_method = all_snippets[0][1]
        best_reward = -1.0
        best_message = "All extractions failed testing"
        
    return best_reward, best_message, best_pass_ratio, best_code, best_method

# ============================================================================
# GRPOTrainer Class
# ============================================================================

class GRPOTrainer:
    def __init__(self, model: PeftModel, tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace, accelerator: Accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.accelerator = accelerator
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
    def train_step(self, batch: Dict[str, Any]) -> tuple:
        self.model.eval()
        device = self.accelerator.device
        
        prompts = batch['prompts']
        tests = batch['tests']
        
        B = len(prompts)
        G = self.args.group_size
        
        # Tokenize queries
        query_tensors = []
        for prompt in prompts:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.args.max_length
            )
            query_tensors.append(tokens.input_ids.squeeze(0).to(device))
            
        # Stack or keep as list. For causal, we expand and generate
        # To avoid OOM, generate one prompt at a time (each generates G outputs)
        all_response_tensors = []
        all_generated_texts = []
        
        gen_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        with torch.no_grad():
            for query_tensor in query_tensors:
                query_batch = query_tensor.unsqueeze(0)
                generated = unwrapped_model.generate(
                    query_batch,
                    num_return_sequences=G,
                    **gen_kwargs
                )
                for gen_seq in generated:
                    response_only = gen_seq[len(query_tensor):]
                    all_response_tensors.append(response_only)
                    all_generated_texts.append(
                        self.tokenizer.decode(response_only, skip_special_tokens=True)
                    )
                    
        # Duplicate prompts to match completions batching
        repeated_prompts = [p for p in prompts for _ in range(G)]
        repeated_tests = [t for t in tests for _ in range(G)]
        
        # Replicate queries
        queries_repeated = [q for q in query_tensors for _ in range(G)]
        
        # Compute rewards via sandboxing
        rewards = []
        raw_rewards_log = []
        pass_ratios = []
        
        for idx, (completion, prompt, test_code) in enumerate(zip(all_generated_texts, repeated_prompts, repeated_tests)):
            reward_val, msg, pass_ratio, _, _ = compute_best_reward_from_all_snippets(completion, test_code)
            
            rewards.append(reward_val)
            raw_rewards_log.append(reward_val)
            pass_ratios.append(pass_ratio)
            
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Group relative advantage normalization
        rewards_reshaped = rewards_tensor.view(B, G)
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True).clamp(min=1e-5)
        advantages = ((rewards_reshaped - mean_rewards) / std_rewards).view(-1)
        
        # 5. Calculate policy log-probs & reference log-probs
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sequence slicing to align precisely
        target_ids = []
        max_target_len = max(len(r) for r in all_response_tensors)
        
        # Pad responses manually for batching
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded_responses = []
        response_masks = []
        for r in all_response_tensors:
            pad_len = max_target_len - len(r)
            padded = torch.cat([r, torch.tensor([pad_id] * pad_len, dtype=torch.long, device=device)])
            mask = torch.cat([torch.ones_like(r), torch.zeros(pad_len, dtype=torch.long, device=device)])
            padded_responses.append(padded)
            response_masks.append(mask)
            
        target_ids = torch.stack(padded_responses)  # [B * G, target_len]
        valid_mask = torch.stack(response_masks)   # [B * G, target_len]
        
        # For causal, input is prompt_ids + response_ids[:-1]
        # Pad prompts to stack them
        max_prompt_len = max(len(q) for q in queries_repeated)
        padded_queries = []
        query_masks = []
        for q in queries_repeated:
            pad_len = max_prompt_len - len(q)
            padded = torch.cat([torch.tensor([pad_id] * pad_len, dtype=torch.long, device=device), q])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), torch.ones_like(q)])
            padded_queries.append(padded)
            query_masks.append(mask)
            
        padded_queries = torch.stack(padded_queries) # [B * G, prompt_len]
        query_masks = torch.stack(query_masks)      # [B * G, prompt_len]
        
        # Target ids for causal predicting are target_ids (the generated tokens)
        # Input to model: [padded_queries, target_ids[:, :-1]]
        inputs = torch.cat([padded_queries, target_ids[:, :-1]], dim=1)
        attention_mask = torch.cat([query_masks, valid_mask[:, :-1]], dim=1)
        
        prefix_len = padded_queries.size(1)
        target_len = target_ids.size(1)
        
        # active policy pass
        policy_outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
        )
        policy_logits = policy_outputs.logits[:, prefix_len - 1 :, :]
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        token_log_probs = torch.gather(policy_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        # reference policy pass (uses the memory-free disable_adapter context manager!)
        if isinstance(unwrapped_model, PeftModel):
            with unwrapped_model.disable_adapter():
                ref_outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits[:, prefix_len - 1 :, :]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                token_ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        else:
            token_ref_log_probs = token_log_probs.detach()
            
        # 6. Compute GRPO loss
        ratio = torch.exp(token_log_probs - token_ref_log_probs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2)
        
        # KL penalty
        kl = torch.exp(token_ref_log_probs - token_log_probs) - (token_ref_log_probs - token_log_probs) - 1.0
        
        # Combine losses
        loss_matrix = (policy_loss + self.args.kl_coef * kl) * valid_mask.float()
        loss = loss_matrix.sum() / valid_mask.float().sum().clamp(min=1.0)
        
        # 7. Backward pass
        self.accelerator.backward(loss)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), float(rewards_tensor.mean().item()), float(np.mean(pass_ratios))

# ============================================================================
# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='raafatabualazm/decompiler-v1')
    parser.add_argument('--dataset_path', default='data/testing/grpo_data.jsonl')
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--kl_coef', type=float, default=0.01)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--output_dir', default='artifacts/qwen-unittest-grpo')
    args = parser.parse_args()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    accelerator = Accelerator(mixed_precision="bf16")
    
    accelerator.print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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
    
    try:
        import flash_attn
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
        
    attn_impl = "flash_attention_2" if has_flash_attn else "sdpa"
    
    accelerator.print(f"Loading model with attn_implementation={attn_impl}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Applied DoRA: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")
    
    # Enable gradient checkpointing for VRAM efficiency
    model.gradient_checkpointing_enable()
    
    # Dataset preparation
    accelerator.print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # Formatting prompt style matching the pretrained templates
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
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'prompts': [item['prompts'] for item in x],
            'tests': [item['tests'] for item in x]
        }
    )
    
    trainer = GRPOTrainer(model, tokenizer, args, accelerator)
    
    accelerator.print("\n=== STARTING UNIT TEST GRPO REINFORCEMENT LEARNING ===")
    step_count = 0
    
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
            
        for batch in pbar:
            loss_val, avg_reward, pass_rate = trainer.train_step(batch)
            
            if accelerator.is_main_process:
                pbar.set_description(
                    f"E{epoch+1} S{step_count+1} | Loss:{loss_val:.3f} | R:{avg_reward:.2f} | Pass:{pass_rate*100:.1f}%"
                )
                
            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break
        if args.max_steps > 0 and step_count >= args.max_steps:
            break
            
    accelerator.print(f"Saving GRPO model to {args.output_dir}...")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save only trainable PEFT adapter weights and tokenizer
        unwrapped = accelerator.unwrap_model(trainer.model)
        state_dict = unwrapped.state_dict()
        trainable_keys = {name for name, param in unwrapped.named_parameters() if param.requires_grad}
        trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}
        
        torch.save(trainable_state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print("GRPO training completed successfully.")

if __name__ == '__main__':
    main()
