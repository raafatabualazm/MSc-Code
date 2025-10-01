from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import json
from codebleu import CodeBLEUCalculator
import statistics
import csv
from torch.cuda.amp import autocast

# Initialize evaluators and storage
dart_eval = CodeBLEUCalculator('dart')
swift_eval = CodeBLEUCalculator('swift')
dart_scores = []
swift_scores = []
scores = []

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
        elif lang.lower() == 'swift':
            code = response_part.split('```swift')[1].split('```')[0].strip()
        else:
            code = response_part.strip()
        
        return code
    except Exception as e:
        print(f"Extraction error: {e}")
        return response

# Batch processing for better GPU utilization
BATCH_SIZE = 4  # Adjust based on your H200 memory (141GB allows larger batches)

data_lines = []
with open('test-set.jsonl') as f:
    data_lines = [json.loads(line) for line in f]

counter = 0
batch_entries = []

def process_batch(batch_entries):
    """Process a batch of inputs"""
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
        max_length=8192  # Adjust based on your needs
    ).to("cuda")
    
    # Generate with optimizations
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=9000,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=0.2,
            top_p=0.99,
            do_sample=True,
            num_beams=5,  # Greedy decoding is faster
        )
    
    # Decode batch
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Process each response
    for idx, (entry, response) in enumerate(zip(batch_entries, responses)):
        lang = entry['language']
        source = entry['source']
        
        code = extract_code(response, lang)
        
        if lang.lower() == 'dart':
            score = dart_eval.compute_codebleu(source, code)['codebleu']
            scores.append(score)
            dart_scores.append(score)
            print(f"Dart score: {score}")
        elif lang.lower() == 'swift':
            score = swift_eval.compute_codebleu(source, code)['codebleu']
            scores.append(score)
            swift_scores.append(score)
            print(f"Swift score: {score}")

# Process data in batches
for entry in data_lines:
    counter += 1
    print(f"Processing {counter}/{len(data_lines)}")
    
    batch_entries.append(entry)
    
    if len(batch_entries) >= BATCH_SIZE:
        process_batch(batch_entries)
        batch_entries = []
        torch.cuda.empty_cache()  # Clear cache between batches

# Process remaining entries
if batch_entries:
    process_batch(batch_entries)

# Statistics calculation
if dart_scores:
    min_dart = min(dart_scores)
    max_dart = max(dart_scores)
    dart_average = statistics.mean(dart_scores)
    dart_stdv = statistics.stdev(dart_scores) if len(dart_scores) > 1 else 0
    
    header = ['Min', 'Max', 'Average', 'Standard_Deviation']
    data_row = [min_dart, max_dart, dart_average, dart_stdv]
    file_name = 'dart_statistics_decompiler-v3.csv'
    
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow(data_row)
    
    print(f"Data successfully written to {file_name}")

# Similar statistics for Swift
if swift_scores:
    min_swift = min(swift_scores)
    max_swift = max(swift_scores)
    swift_average = statistics.mean(swift_scores)
    swift_stdv = statistics.stdev(swift_scores) if len(swift_scores) > 1 else 0
    
    file_name = 'swift_statistics_decompiler-v3.csv'
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow([min_swift, max_swift, swift_average, swift_stdv])
    
    print(f"Swift data successfully written to {file_name}")
