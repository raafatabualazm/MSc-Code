import json
import os
from openai import OpenAI


inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}. 
Please convert the following assembly code to idiomatic and clear {} code. 

### Assembly:
{}
"""

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY before running this script.")
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

input_file = 'data/intermediate/failed_reasoning7.jsonl'
output_file = 'data/intermediate/all_data_new_reason5.jsonl'
failed_file = 'data/intermediate/failed_reasoning8.jsonl'

try:
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile, open(failed_file, 'w') as failfile:
        for line in infile:
            try:
                data = json.loads(line)
                prompt = inference_prompt_style.format(data["assembly"], data["language"], data["language"]) 
                if prompt:
                    response = client.chat.completions.create(
                        model="moonshotai/kimi-k2-thinking",
                        messages=[{ "role": "system", "content": "You are a helpful assistant with very high analysis and reverse engineering capabilities." }, {"role": "user", "content": prompt}],
                        temperature=0.6,
                        max_tokens=16384,
                        top_p=1.0,
                        logprobs=True,
                        top_logprobs=2,
                        extra_body= {
                        "provider": {
                            "only": ["nebius", "parasail"]
                        }
                        }
                    
                    )
                    logprobs_data = response.choices[0].logprobs.content
                    data['logprobs'] = [
                            item.model_dump() for item in logprobs_data
                        ]
                else:
                    print("No 'text' field found in JSON object; skipping API call.")
                outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
            except Exception as e:
                print(f"Error processing line: {e}")
                failfile.write(line)
except FileNotFoundError:
    print(f"Input file {input_file} not found.")

