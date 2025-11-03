from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator
import json
import csv
import re
import os
import tempfile
import subprocess
from torch.cuda.amp import autocast
from collections import defaultdict

# ============================================================================
# TEST SCRIPT WITH ACTUAL EXECUTION - DART/SWIFT COMPILATION & TEST RESULTS
# ============================================================================

print("🚀 Decompiler Test Script with Execution-Based Evaluation")
print("=" * 80)

# Initialize accelerator
accelerator = Accelerator()

# Optimized CUDA settings for H200
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.cuda.set_device(accelerator.local_process_index)

model_dir = "decompiler-v3"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

model.config.use_cache = True
model.eval()

# Compile model with optimizations
try:
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
except Exception as e:
    print(f"Compilation warning: {e}")
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except:
        pass

# ============================================================================
# DART SANDBOX EXECUTION
# ============================================================================

def run_dart_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    """Execute Dart code with tests and return result"""
    if not solution_code.strip():
        return False, "Error: Empty solution code", 0, 0
    
    # Check for main() violation
    if 'void main(' in solution_code or 'main()' in solution_code:
        return False, "Error: Solution should only contain the function, not main()", 0, 0
    
    # Separate imports from function
    lines = solution_code.split('\n')
    imports = []
    function_lines = []
    
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith('import ') or stripped.startswith('export ') or
            stripped.startswith('@pragma(') or stripped.startswith('library ') or
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
            
            # Count test cases (approximate from test code)
            total_tests = test_code.count('expect(')
            
            if test_proc.returncode == 0:
                return True, "✓ All tests passed", total_tests, total_tests
            elif "Error:" in test_proc.stderr or "Error:" in test_proc.stdout:
                error_msg = (test_proc.stderr or test_proc.stdout)[:200]
                return False, f"Compilation Error: {error_msg}", total_tests, 0
            else:
                # Try to count which tests passed by looking at output
                passed_tests = 0
                error_msg = (test_proc.stderr or test_proc.stdout)[:200]
                return False, f"Test Failure: {error_msg}", total_tests, passed_tests
                
    except subprocess.TimeoutExpired:
        total_tests = test_code.count('expect(')
        return False, "⏱ Timeout", total_tests, 0
    except Exception as e:
        total_tests = test_code.count('expect(')
        return False, f"Sandbox Error: {str(e)}", total_tests, 0

# ============================================================================
# SWIFT SANDBOX EXECUTION
# ============================================================================

def run_swift_sandbox(solution_code: str, test_code: str, timeout: int = 10) -> tuple:
    """Execute Swift code with tests and return result"""
    if not solution_code.strip():
        return False, "Error: Empty solution code", 0, 0
    
    # For Swift, combine solution and test code
    full_code = solution_code + "\n\n" + test_code
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filepath = os.path.join(temp_dir, 'temp_test.swift')
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(full_code)
            
            # Compile Swift
            compile_proc = subprocess.run(
                ['swiftc', test_filepath, '-o', os.path.join(temp_dir, 'test_binary')],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            total_tests = test_code.count('assert(') + test_code.count('XCTAssert')
            
            if compile_proc.returncode != 0:
                error_msg = compile_proc.stderr[:200]
                return False, f"Compilation Error: {error_msg}", total_tests, 0
            
            # Run the compiled binary
            run_proc = subprocess.run(
                [os.path.join(temp_dir, 'test_binary')],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if run_proc.returncode == 0:
                return True, "✓ All tests passed", total_tests, total_tests
            else:
                error_msg = (run_proc.stderr or run_proc.stdout)[:200]
                return False, f"Test Failure: {error_msg}", total_tests, 0
                
    except subprocess.TimeoutExpired:
        total_tests = test_code.count('assert(') + test_code.count('XCTAssert')
        return False, "⏱ Timeout", total_tests, 0
    except Exception as e:
        total_tests = test_code.count('assert(') + test_code.count('XCTAssert')
        return False, f"Sandbox Error: {str(e)}", total_tests, 0

# ============================================================================
# CODE EXTRACTION - Multi-pattern approach
# ============================================================================

def extract_all_code_snippets(completion: str, lang: str) -> list:
    """
    Extract ALL potential code snippets from completion.
    Returns: List of (code, extraction_method) tuples
    """
    snippets = []
    lang_lower = lang.lower()
    
    # Pattern 1: Language-specific fenced blocks
    pattern = rf"```{lang_lower}\s*(.*?)\s*```"
    for match in re.finditer(pattern, completion, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if code:
            snippets.append((code, f"{lang_lower}_fenced"))
    
    # Pattern 2: Generic fenced blocks
    for match in re.finditer(r"```\s*(.*?)\s*```", completion, re.DOTALL):
        code = match.group(1).strip()
        if code:
            # Basic check if it looks like valid code
            if not any(snippet[0] == code for snippet in snippets):
                snippets.append((code, "generic_fenced"))
    
    # Pattern 3: Find function signatures
    if lang_lower == 'dart':
        pattern = r"(List<[\w<>, ]+>|String|int|double|bool|void|Map<[\w<>, ]+>|Set<[\w<>, ]+>)\s+\w+\s*\([^)]*\)\s*\{"
    else:  # Swift
        pattern = r"func\s+\w+\s*\([^)]*\)\s*(?:->\s*[\w<>, ]+)?\s*\{"
    
    for match in re.finditer(pattern, completion, re.DOTALL):
        start_idx = match.start()
        code_part = completion[start_idx:]
        
        # Extract balanced braces
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
            if code and not any(snippet[0] == code for snippet in snippets):
                snippets.append((code, "signature_search"))
    
    # Pattern 4: Response delimiter
    if "### Response:" in completion:
        response_part = completion.split("### Response:")[-1].strip()
        # Try to extract code from response section
        if not response_part.startswith("```"):
            # Might be raw code
            lines = response_part.split('\n')
            code_lines = []
            for line in lines:
                if line.strip() and not line.strip().startswith(('###', '**', 'Note:', 'Explanation')):
                    code_lines.append(line)
                elif code_lines:
                    break
            if code_lines:
                code = '\n'.join(code_lines).strip()
                if code and len(code) > 30 and not any(snippet[0] == code for snippet in snippets):
                    snippets.append((code, "response_section"))
    
    return snippets

def extract_code(response: str, lang: str) -> tuple:
    """
    Extract the best code snippet from response.
    Returns: (code, extraction_method, num_attempts)
    """
    all_snippets = extract_all_code_snippets(response, lang)
    
    if not all_snippets:
        # Fallback to entire response if it looks like code
        if len(response.strip()) > 50:
            return response.strip(), "raw_text", 0
        return None, "none", 0
    
    # Return the first (usually best) snippet
    return all_snippets[0][0], all_snippets[0][1], len(all_snippets)

# ============================================================================
# TESTING WITH EXECUTION
# ============================================================================

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

BATCH_SIZE = 4

# Load test data
data_lines = []
with open('test-set.jsonl') as f:
    data_lines = [json.loads(line) for line in f]

# Statistics tracking
stats = {
    'total_files': 0,
    'compiled_successfully': 0,
    'compilation_failed': 0,
    'total_test_cases': 0,
    'passed_test_cases': 0,
    'by_language': defaultdict(lambda: {
        'total': 0,
        'compiled': 0,
        'total_tests': 0,
        'passed_tests': 0
    }),
    'extraction_methods': defaultdict(int),
    'detailed_results': []
}

def process_batch(batch_entries):
    """Process a batch of inputs with execution-based evaluation"""
    if not batch_entries:
        return
    
    prompts = [
        inference_prompt_style.format(entry['language'], entry['language'], entry['assembly']) 
        + tokenizer.eos_token
        for entry in batch_entries
    ]
    
    # Batch tokenization
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to("cuda")
    
    # Generate
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
        )
    
    # Decode batch
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Process each response with execution
    for idx, (entry, response) in enumerate(zip(batch_entries, responses)):
        lang = entry['language']
        test_code = entry.get('tests', '')
        filename = entry.get('filename', f'unknown_{stats["total_files"]}')
        
        # Extract code
        code, extraction_method, num_attempts = extract_code(response, lang)
        
        stats['extraction_methods'][extraction_method] += 1
        
        # Execute tests
        if code:
            if lang.lower() == 'dart':
                compiled, message, total_tests, passed_tests = run_dart_sandbox(code, test_code)
            elif lang.lower() == 'swift':
                compiled, message, total_tests, passed_tests = run_swift_sandbox(code, test_code)
            else:
                compiled, message, total_tests, passed_tests = False, "Unsupported language", 0, 0
        else:
            compiled, message, total_tests, passed_tests = False, "No code extracted", 0, 0
        
        # Update statistics
        stats['total_files'] += 1
        stats['total_test_cases'] += total_tests
        stats['passed_test_cases'] += passed_tests
        
        lang_stats = stats['by_language'][lang]
        lang_stats['total'] += 1
        lang_stats['total_tests'] += total_tests
        lang_stats['passed_tests'] += passed_tests
        
        if compiled:
            stats['compiled_successfully'] += 1
            lang_stats['compiled'] += 1
            status = "✓ PASSED"
        else:
            stats['compilation_failed'] += 1
            status = "✗ FAILED"
        
        # Store detailed result
        result = {
            'filename': filename,
            'language': lang,
            'compiled': compiled,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
            'message': message,
            'extraction_method': extraction_method,
            'num_snippets_found': num_attempts
        }
        stats['detailed_results'].append(result)
        
        # Print progress
        print(f"{status} [{stats['total_files']}/{len(data_lines)}] {filename} ({lang}): "
              f"{passed_tests}/{total_tests} tests passed | Method: {extraction_method}")

# Process data in batches
counter = 0
batch_entries = []

print(f"\nProcessing {len(data_lines)} files...")
print("=" * 80)

for entry in data_lines:
    counter += 1
    batch_entries.append(entry)
    
    if len(batch_entries) >= BATCH_SIZE:
        process_batch(batch_entries)
        batch_entries = []
        torch.cuda.empty_cache()

# Process remaining entries
if batch_entries:
    process_batch(batch_entries)

# ============================================================================
# GENERATE COMPREHENSIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL STATISTICS")
print("=" * 80)

# Overall statistics
print(f"\nOVERALL:")
print(f"  Total Files Processed: {stats['total_files']}")
print(f"  Successfully Compiled: {stats['compiled_successfully']} ({stats['compiled_successfully']/stats['total_files']*100:.1f}%)")
print(f"  Compilation Failed:    {stats['compilation_failed']} ({stats['compilation_failed']/stats['total_files']*100:.1f}%)")
print(f"  Total Test Cases:      {stats['total_test_cases']}")
print(f"  Passed Test Cases:     {stats['passed_test_cases']} ({stats['passed_test_cases']/stats['total_test_cases']*100:.1f}%)")

# Per-language statistics
print(f"\nBY LANGUAGE:")
for lang, lang_stats in stats['by_language'].items():
    print(f"  {lang}:")
    print(f"    Files:              {lang_stats['total']}")
    print(f"    Compiled:           {lang_stats['compiled']} ({lang_stats['compiled']/lang_stats['total']*100:.1f}%)")
    print(f"    Total Tests:        {lang_stats['total_tests']}")
    print(f"    Passed Tests:       {lang_stats['passed_tests']} ({lang_stats['passed_tests']/lang_stats['total_tests']*100:.1f}%)")

# Extraction method statistics
print(f"\nEXTRACTION METHODS:")
for method, count in sorted(stats['extraction_methods'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {method}: {count} ({count/stats['total_files']*100:.1f}%)")

print("=" * 80)

# ============================================================================
# SAVE DETAILED RESULTS TO CSV
# ============================================================================

# Overall statistics CSV
with open('execution_statistics_overall.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value', 'Percentage'])
    writer.writerow(['Total Files', stats['total_files'], '100.0%'])
    writer.writerow(['Compiled Successfully', stats['compiled_successfully'], 
                    f"{stats['compiled_successfully']/stats['total_files']*100:.1f}%"])
    writer.writerow(['Compilation Failed', stats['compilation_failed'],
                    f"{stats['compilation_failed']/stats['total_files']*100:.1f}%"])
    writer.writerow(['Total Test Cases', stats['total_test_cases'], ''])
    writer.writerow(['Passed Test Cases', stats['passed_test_cases'],
                    f"{stats['passed_test_cases']/stats['total_test_cases']*100:.1f}%"])

print(f"\n✓ Overall statistics saved to: execution_statistics_overall.csv")

# Per-language statistics CSV
for lang, lang_stats in stats['by_language'].items():
    filename = f'execution_statistics_{lang.lower()}.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value', 'Percentage'])
        writer.writerow(['Total Files', lang_stats['total'], '100.0%'])
        writer.writerow(['Compiled Successfully', lang_stats['compiled'],
                        f"{lang_stats['compiled']/lang_stats['total']*100:.1f}%"])
        writer.writerow(['Total Test Cases', lang_stats['total_tests'], ''])
        writer.writerow(['Passed Test Cases', lang_stats['passed_tests'],
                        f"{lang_stats['passed_tests']/lang_stats['total_tests']*100:.1f}%"])
    print(f"✓ {lang} statistics saved to: {filename}")

# Detailed per-file results CSV
with open('execution_results_detailed.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Language', 'Compiled', 'Total_Tests', 'Passed_Tests', 
                     'Pass_Rate', 'Extraction_Method', 'Snippets_Found', 'Message'])
    
    for result in stats['detailed_results']:
        writer.writerow([
            result['filename'],
            result['language'],
            'Yes' if result['compiled'] else 'No',
            result['total_tests'],
            result['passed_tests'],
            f"{result['pass_rate']:.1f}%",
            result['extraction_method'],
            result['num_snippets_found'],
            result['message'][:100]  # Truncate long messages
        ])

print(f"✓ Detailed per-file results saved to: execution_results_detailed.csv")

# Save extraction method statistics
with open('extraction_methods_statistics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Extraction_Method', 'Count', 'Percentage'])
    for method, count in sorted(stats['extraction_methods'].items(), key=lambda x: x[1], reverse=True):
        writer.writerow([method, count, f"{count/stats['total_files']*100:.1f}%"])

print(f"✓ Extraction methods saved to: extraction_methods_statistics.csv")

print("\n" + "=" * 80)
print("🎉 TESTING COMPLETE!")
print("=" * 80)