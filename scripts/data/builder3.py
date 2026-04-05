import json
import time
import random
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

API_KEY = "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"
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

input_file = "final_analysis.jsonl"
output_file = "data_swift_reason_new.jsonl"
failed_file = "failed_reasoning4.jsonl"

MAX_RETRIES = 8
INITIAL_BACKOFF = 3  # seconds
MAX_BACKOFF = 60     # seconds


def is_retryable_429(exc: Exception) -> bool:
    """
    Returns True only for the temporary upstream/provider 429 case.
    """
    msg = str(exc)
    return (
        "429" in msg
        and (
            "temporarily rate-limited upstream" in msg
            or "Provider returned error" in msg
            or "Please retry shortly" in msg
        )
    )


def request_with_retry(prompt: str):
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2-speciale",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with very high analysis and reverse engineering capabilities."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1.0,
                max_tokens=81000,
                top_p=1.0,
                extra_body={"reasoning": {"enabled": True}},
            )
            return response

        except Exception as e:
            if is_retryable_429(e) and attempt < MAX_RETRIES:
                sleep_time = min(backoff, MAX_BACKOFF) + random.uniform(0, 1.5)
                print(
                    f"[Retryable 429] Attempt {attempt}/{MAX_RETRIES} failed. "
                    f"Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
                backoff *= 2
                continue

            raise


try:
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "a", encoding="utf-8") as outfile, \
         open(failed_file, "a", encoding="utf-8") as failfile:

        for line_num, line in enumerate(infile, start=1):
            try:
                data = json.loads(line)
                prompt = inference_prompt_style.format(
                    data["language"],
                    data["language"],
                    data["assembly"]
                )

                if prompt:
                    response = request_with_retry(prompt)
                    reasoning = response.choices[0].message.content
                    data["reasoning"] = reasoning
                else:
                    print(f"Line {line_num}: Empty prompt; skipping API call.")

                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                outfile.flush()

            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON")
                failfile.write(line)
                failfile.flush()

            except Exception as e:
                print(f"Line {line_num}: Error processing line: {e}")
                failfile.write(line)
                failfile.flush()

except FileNotFoundError:
    print(f"Input file {input_file} not found.")