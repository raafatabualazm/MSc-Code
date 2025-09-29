from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import Accelerator, PartialState
import json
from codebleu import CodeBLEUCalculator
import statistics
import csv

# ---------------- Setup ----------------
dart_eval = CodeBLEUCalculator('dart')
swift_eval = CodeBLEUCalculator('swift')

accelerator = Accelerator()
state = PartialState()
rank = state.process_index
world = state.num_processes

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model_dir = "decompiler-v3"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},  # one GPU per process
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    # quantization_config=bnb_config,
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

# Local (per-rank) results
local_dart_scores = []
local_swift_scores = []

def handle(line: str):
    entry = json.loads(line)
    assembly = entry['assembly']
    lang = entry['language']
    source = entry['source']

    inputs = tokenizer(
        [inference_prompt_style.format(lang, lang, assembly) + tokenizer.eos_token],
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=9000,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=0.2,
            top_p=0.99,
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    text = response[0]

    # Try to extract code fenced block; fall back to everything after the header
    if lang.lower() == 'dart':
        try:
            code = text.split("### Response:")[1].split('```dart')[1].split('```')[0].strip()
        except Exception:
            code = text.split("### Response:", 1)[-1].strip()
        score = dart_eval.compute_codebleu(source, code)['codebleu']
        local_dart_scores.append(score)
        print(f"[rank {rank}] dart score: {score}")

    elif lang.lower() == 'swift':
        try:
            code = text.split("### Response:")[1].split('```swift')[1].split('```')[0].strip()
        except Exception:
            code = text.split("### Response:", 1)[-1].strip()
        score = swift_eval.compute_codebleu(source, code)['codebleu']
        local_swift_scores.append(score)
        print(f"[rank {rank}] swift score: {score}")

# ---------------- Sharded reading ----------------
with open('dart_all.jsonl', 'r', encoding='utf-8', errors='replace') as data:
    for i, line in enumerate(data):
        if i % world == rank:
            handle(line)

accelerator.wait_for_everyone()

# ---------------- Gather & merge on main ----------------
all_dart_lists = accelerator.gather_object(local_dart_scores)   # list-of-lists
all_swift_lists = accelerator.gather_object(local_swift_scores)

if accelerator.is_main_process:
    # Flatten
    dart_scores = [s for sub in all_dart_lists for s in sub]
    swift_scores = [s for sub in all_swift_lists for s in sub]
    combined_scores = dart_scores + swift_scores

    # Helper to write stats if we have data
    def write_stats(scores, csv_name_prefix):
        if not scores:
            print(f"No {csv_name_prefix} scores collected.")
            return
        # stdev needs at least 2 items
        min_v = min(scores)
        max_v = max(scores)
        avg_v = statistics.mean(scores)
        std_v = statistics.stdev(scores) if len(scores) > 1 else 0.0

        header = ['Min', 'Max', 'Average', 'Standard_Deviation', 'Count']
        row = [min_v, max_v, avg_v, std_v, len(scores)]
        file_name = f'{csv_name_prefix}_statistics-decompiler-v3.csv'
        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(row)
        print(f"Data successfully written to {file_name}")

    write_stats(dart_scores, "dart")
    write_stats(swift_scores, "swift")
    write_stats(combined_scores, "combined")
    
accelerator.wait_for_everyone()
