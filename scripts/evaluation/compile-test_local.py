from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import json
from codebleu import CodeBLEUCalculator
import statistics
import csv
from torch.cuda.amp import autocast
import subprocess
import tempfile
import os

# Initialize evaluators and storage
dart_eval = CodeBLEUCalculator('dart')
dart_scores = []
compilation_success_count = 0
compilation_failure_count = 0
total_attempts = 0

accelerator = Accelerator()

# Optimized CUDA settings for H200
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune kernels
torch.set_float32_matmul_precision("high")

# Enable CUDA graphs for faster inference
torch.cuda.set_device(accelerator.local_process_index)

model_dir = "decompiler-v3"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},  
    torch_dtype=torch.bfloat16,  # Use torch_dtype instead of dtype
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

# Pre-compile regex patterns if needed
def extract_code(response, lang):
    """Optimized code extraction"""
    try:
        parts = response.split("### Response:")
        if len(parts) < 2:
            return parts[0] if parts else ""
        
        response_part = parts[1]
        
        if lang.lower() == 'dart':
            try:
                code = response_part.split('```dart')[1].split('```')[0].strip()
            except:
                code = response_part.strip()
        else:
            code = response_part.strip()
        
        return code
    except Exception as e:
        print(f"Extraction error: {e}")
        return response

def compile_dart_code(code):
    """
    Try to compile Dart code using dart compile aot-snapshot
    Returns True if compilation succeeds, False otherwise
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dart_file = os.path.join(tmpdir, "test.dart")
        snapshot_file = os.path.join(tmpdir, "test.aot")
        
        # Wrap code in a main function if it doesn't have one
        if "void main()" not in code and "main(" not in code:
            wrapped_code = f"void main() {{\n{code}\n}}"
        else:
            wrapped_code = code
        
        try:
            # Write code to file
            with open(dart_file, 'w') as f:
                f.write(wrapped_code)
            
            # Try to compile
            result = subprocess.run(
                ['dart', 'compile', 'aot-snapshot', dart_file, '-o', snapshot_file],
                capture_output=True,
                timeout=30,
                text=True
            )
            
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("Compilation timeout")
            return False
        except Exception as e:
            print(f"Compilation error: {e}")
            return False

# Constants
K = 5  # Number of generations per input
BATCH_SIZE = 1  # Process one at a time since we need K generations per input

data_lines = []
with open('data/datasets/test-set.jsonl') as f:
    data_lines = [json.loads(line) for line in f]

counter = 0

def process_entry(entry):
    """Process a single entry with K generations"""
    global compilation_success_count, compilation_failure_count, total_attempts
    
    lang = entry['language']
    source = entry['source']
    
    # Only process Dart code
    if lang.lower() != 'dart':
        print(f"Skipping non-Dart language: {lang}")
        return
    
    prompt = inference_prompt_style.format(lang, lang, entry['assembly']) + tokenizer.eos_token
    
    # Tokenization
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to("cuda")
    
    valid_scores = []
    
    # Generate K times
    for k in range(K):
        print(f"  Generation {k+1}/{K}")
        total_attempts += 1
        
        # Generate
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=9000,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                temperature=0.7,  # Increased for diversity
                top_p=0.95,
                do_sample=True,
                num_beams=1,  # Sampling instead of beam search for diversity
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = extract_code(response, lang)
        
        # Try to compile
        if compile_dart_code(code):
            print(f"    ✓ Compilation succeeded")
            compilation_success_count += 1
            
            # Calculate CodeBLEU for successful compilation
            score = dart_eval.compute_codebleu(source, code)['codebleu']
            valid_scores.append(score)
            print(f"    CodeBLEU: {score:.4f}")
        else:
            print(f"    ✗ Compilation failed")
            compilation_failure_count += 1
    
    # Take max score from valid compilations
    if valid_scores:
        max_score = max(valid_scores)
        dart_scores.append(max_score)
        print(f"  Max CodeBLEU: {max_score:.4f} ({len(valid_scores)}/{K} compiled successfully)")
    else:
        print(f"  No successful compilations (0/{K})")

# Process data
for entry in data_lines:
    counter += 1
    print(f"\nProcessing {counter}/{len(data_lines)}")
    
    process_entry(entry)
    torch.cuda.empty_cache()  # Clear cache between entries

# Statistics calculation
print("\n" + "="*50)
print("FINAL STATISTICS")
print("="*50)

print(f"\nCompilation Statistics:")
print(f"  Total generation attempts: {total_attempts}")
print(f"  Successful compilations: {compilation_success_count} ({compilation_success_count/total_attempts*100:.2f}%)")
print(f"  Failed compilations: {compilation_failure_count} ({compilation_failure_count/total_attempts*100:.2f}%)")

if dart_scores:
    min_dart = min(dart_scores)
    max_dart = max(dart_scores)
    dart_average = statistics.mean(dart_scores)
    dart_stdv = statistics.stdev(dart_scores) if len(dart_scores) > 1 else 0
    
    print(f"\nCodeBLEU Statistics (max scores from successful compilations):")
    print(f"  Samples with at least one successful compilation: {len(dart_scores)}")
    print(f"  Min: {min_dart:.4f}")
    print(f"  Max: {max_dart:.4f}")
    print(f"  Average: {dart_average:.4f}")
    print(f"  Std Dev: {dart_stdv:.4f}")
    
    # Write CodeBLEU statistics
    header = ['Min', 'Max', 'Average', 'Standard_Deviation', 'Samples_With_Success', 'Total_Samples']
    data_row = [min_dart, max_dart, dart_average, dart_stdv, len(dart_scores), len(data_lines)]
    file_name = 'results/statistics/dart_statistics_decompiler-v3_compiled.csv'
    
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow(data_row)
    
    print(f"\nCodeBLEU statistics written to {file_name}")
    
    # Write compilation statistics
    comp_header = ['Total_Attempts', 'Successful_Compilations', 'Failed_Compilations', 
                   'Success_Rate', 'K_Value']
    comp_data = [total_attempts, compilation_success_count, compilation_failure_count,
                 compilation_success_count/total_attempts if total_attempts > 0 else 0, K]
    comp_file = 'dart_compilation_statistics_decompiler-v3.csv'
    
    with open(comp_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(comp_header)
        writer.writerow(comp_data)
    
    print(f"Compilation statistics written to {comp_file}")
else:
    print("\nNo successful compilations across all samples!")


