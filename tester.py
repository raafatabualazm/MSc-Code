from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import json
from codebleu import CodeBLEUCalculator
import statistics
import csv

dart_eval = CodeBLEUCalculator('dart')
swift_eval = CodeBLEUCalculator('swift')
dart_scores = []
swift_scores = []
scores = []

accelerator = Accelerator()

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model_dir = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map={"": accelerator.local_process_index},  
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
    # quantization_config=bnb_config            
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
data = open('dart_all.jsonl')
counter = 0
for line in data:
    entry = json.loads(line)
    counter +=1
    print(counter)
    assembly = entry['assembly']
    lang = entry['language']
    source = entry['source']
    inputs = tokenizer(
        [inference_prompt_style.format(lang, lang, assembly) + tokenizer.eos_token],
        return_tensors="pt"
    ).to("cuda")


    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=6000,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        temperature=0.2,
        top_p=0.99,
    )
    code = ""
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if lang.lower() == 'dart':
        try:
            code = response[0].split("### Response:")[1].split('```dart')[1].split('```')[0].strip()
        except:
            code = response[0].split("### Response:")[1]
        score = dart_eval.compute_codebleu(source, code)
        score = score['codebleu']
        scores.append(score)
        dart_scores.append(score)
        print(score)
    elif lang.lower() == 'swift':
        code = response[0].split("### Response:")[1].split('```swift')[1].split('```')[0].strip()
        score = swift_eval.compute_codebleu(source, code)
        score = score['codebleu']
        scores.append(score)
        swift_scores.append(score)
        print(score)
    print(code)

min_dart = min(dart_scores)
max_dart = max(dart_scores)
dart_average = statistics.mean(dart_scores)
dart_stdv = statistics.stdev(dart_scores)
header = ['Min', 'Max', 'Average', 'Standard_Deviation']
data_row = [min_dart, max_dart, dart_average, dart_stdv]
file_name = 'dart_statistics-Qwen3-4B-Thinking-2507.csv'
with open(file_name, 'w', newline='') as csvfile:
    # Create a writer object
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(header)
    
    # Write the data row
    writer.writerow(data_row)

print(f"Data successfully written to {file_name}")