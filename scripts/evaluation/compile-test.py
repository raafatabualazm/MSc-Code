from openai import OpenAI
import json
from math import comb
from codebleu import CodeBLEUCalculator
import statistics
import csv
import subprocess
import tempfile
import os
import argparse
import sys

# --- Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name/path served by the API")
parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL")
parser.add_argument("--api-key", type=str, default="EMPTY", help="API key (use EMPTY for local vLLM/TGI)")
parser.add_argument("--test-set", type=str, default="data/datasets/test-set2.jsonl", help="Path to test set JSONL")
parser.add_argument("--k", type=int, default=5, help="Number of generations per input (compile@K)")
parser.add_argument("--batch-size", type=int, default=8, help="Number of completions per API call (n parameter)")
parser.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens to generate")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint JSONL path (default: auto from model name)")
parser.add_argument("--max-prompt-tokens", type=int, default=60000, help="Max prompt tokens before skipping a sample")
args = parser.parse_args()

MODEL_NAME = args.model
K = args.k
BATCH_SIZE = args.batch_size
MAX_PROMPT_TOKENS = args.max_prompt_tokens

# Sanitize model name for filenames
model_slug = MODEL_NAME.replace("/", "_").replace("\\", "_").replace(" ", "_")

# Checkpoint file for resume support
CHECKPOINT_FILE = args.checkpoint or f"results/cache/checkpoint_{model_slug}.jsonl"

# Initialize OpenAI-compatible client
client = OpenAI(base_url=args.base_url, api_key=args.api_key)

# Initialize evaluators and storage
dart_eval = CodeBLEUCalculator('dart')
# Per-sample results: list of dicts with n, c, codebleu scores
sample_results = []
total_attempts = 0
total_successes = 0
total_failures = 0

# Lazy-loaded tokenizer for prompt length checking
_tokenizer = None


def get_tokenizer():
    """Lazy-load tokenizer for prompt length estimation."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer for {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return _tokenizer


class APIFailure(Exception):
    """Raised when an API call fails, to trigger early termination."""
    pass


SYSTEM_PROMPT = (
    "You are a reverse engineering expert with advanced knowledge in assembly and {lang}. "
    "Please convert the following assembly code to idiomatic and clear {lang} code. "
    "Think carefully about the task and create a step-by-step chain of thoughts to ensure "
    "a logical and accurate response. "
    "Return your final answer inside a ```{lang_lower}``` code block."
)

USER_PROMPT = "Convert the following assembly to idiomatic {lang} code:\n\n```asm\n{assembly}\n```"


def load_checkpoint() -> list[dict]:
    """Load existing checkpoint results. Returns list of per-sample result dicts."""
    if not os.path.exists(CHECKPOINT_FILE):
        return []
    results = []
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_checkpoint_entry(result: dict):
    """Append a single sample result to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')


def compute_compile_at_k(n: int, c: int, k: int) -> float:
    """Unbiased compile@k estimator (Chen et al., 2021).
    
    compile@k = 1 - C(n-c, k) / C(n, k)
    
    n: total generations for this sample
    c: number that compiled successfully
    k: the k we are estimating for
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def extract_code(response: str, lang: str) -> str:
    """Extract code from a markdown-fenced response."""
    try:
        tag = lang.lower()
        if f"```{tag}" in response:
            code = response.split(f"```{tag}")[1].split("```")[0].strip()
            return code
        if "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
            return code
        return response.strip()
    except Exception as e:
        print(f"Extraction error: {e}")
        return response.strip()


def compile_dart_code(code: str) -> bool:
    """Try to compile Dart code using dart compile aot-snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dart_file = os.path.join(tmpdir, "test.dart")
        snapshot_file = os.path.join(tmpdir, "test.aot")

        if "void main()" not in code and "main(" not in code:
            wrapped_code = f"void main() {{\n{code}\n}}"
        else:
            wrapped_code = code

        try:
            with open(dart_file, 'w', encoding='utf-8') as f:
                f.write(wrapped_code)

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


def generate_completions_batched(lang: str, assembly: str, num: int) -> list[str]:
    """Generate `num` completions, issuing API calls with n=BATCH_SIZE at a time.
    
    Raises APIFailure on any API error so the caller can stop and resume later.
    """
    all_texts = []
    remaining = num
    while remaining > 0:
        batch_n = min(BATCH_SIZE, remaining)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(lang=lang, lang_lower=lang.lower()),
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(lang=lang, assembly=assembly),
                    },
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n=batch_n,
            )
            for choice in response.choices:
                all_texts.append(choice.message.content)
        except Exception as e:
            raise APIFailure(f"API error (batch of {batch_n}): {e}") from e
        remaining -= batch_n
    return all_texts


def process_entry(entry: dict) -> bool:
    """Process a single entry with K generations in batches.
    
    Returns True if completed successfully, False if interrupted by API failure.
    """
    global total_attempts, total_successes, total_failures

    lang = entry['language']
    source = entry['source']

    if lang.lower() != 'dart':
        print(f"Skipping non-Dart language: {lang}")
        return True

    # Skip samples with assembly too long for the model context
    tokenizer = get_tokenizer()
    prompt_tokens = len(tokenizer.encode(entry['assembly']))
    if prompt_tokens > MAX_PROMPT_TOKENS:
        print(f"  ⚠ Skipping: assembly too long ({prompt_tokens} tokens > {MAX_PROMPT_TOKENS} limit)")
        result = {
            'n': K,
            'c': 0,
            'compile_at_1': 0.0,
            'compile_at_k': 0.0,
            'codebleu_scores': [],
            'max_codebleu': None,
            'skipped': True,
            'skip_reason': f'prompt_too_long ({prompt_tokens} tokens)',
        }
        sample_results.append(result)
        save_checkpoint_entry(result)
        total_attempts += K
        total_failures += K
        return True

    # Generate all K completions in batches of BATCH_SIZE
    print(f"  Generating {K} completions (batch_size={BATCH_SIZE})...")
    try:
        responses = generate_completions_batched(lang, entry['assembly'], K)
    except APIFailure as e:
        print(f"  ✗ {e}")
        return False

    n = len(responses)  # should be K
    c = 0               # successful compilations
    codebleu_scores = []

    for i, response_text in enumerate(responses):
        total_attempts += 1
        if response_text is None:
            print(f"    [{i+1}/{K}] ✗ Empty response")
            total_failures += 1
            continue

        code = extract_code(response_text, lang)

        if compile_dart_code(code):
            c += 1
            total_successes += 1
            score = dart_eval.compute_codebleu(source, code)['codebleu']
            codebleu_scores.append(score)
            print(f"    [{i+1}/{K}] ✓ Compiled  CodeBLEU={score:.4f}")
        else:
            total_failures += 1
            print(f"    [{i+1}/{K}] ✗ Compilation failed")

    # Compute compile@k for multiple k values
    compile_at_1 = compute_compile_at_k(n, c, 1)
    compile_at_k = compute_compile_at_k(n, c, K)

    result = {
        'n': n,
        'c': c,
        'compile_at_1': compile_at_1,
        'compile_at_k': compile_at_k,
        'codebleu_scores': codebleu_scores,
        'max_codebleu': max(codebleu_scores) if codebleu_scores else None,
    }

    sample_results.append(result)
    save_checkpoint_entry(result)

    print(f"  Compiled: {c}/{n}  compile@1={compile_at_1:.4f}  compile@{K}={compile_at_k:.4f}"
          + (f"  max_CodeBLEU={max(codebleu_scores):.4f}" if codebleu_scores else ""))

    return True


# --- Main ---
data_lines = []
with open(args.test_set, 'r', encoding='utf-8') as f:
    data_lines = [json.loads(line) for line in f]

# Load checkpoint and determine where to resume
checkpoint_results = load_checkpoint()
start_index = len(checkpoint_results)

if start_index > 0:
    print(f"Resuming from checkpoint: {start_index}/{len(data_lines)} samples already completed.")
    # Restore accumulated state from checkpoint
    sample_results = checkpoint_results
    for r in checkpoint_results:
        total_attempts += r['n']
        total_successes += r['c']
        total_failures += r['n'] - r['c']
else:
    print(f"Starting fresh ({len(data_lines)} samples).")

stopped_early = False
for counter, entry in enumerate(data_lines, 1):
    if counter <= start_index:
        continue  # skip already-checkpointed samples

    print(f"\nProcessing {counter}/{len(data_lines)}")
    success = process_entry(entry)
    if not success:
        print(f"\n⚠ Stopped at sample {counter}/{len(data_lines)} due to API failure.")
        print(f"  Re-run the same command to resume from sample {counter}.")
        stopped_early = True
        break

# --- Statistics ---
print("\n" + "=" * 50)
print("FINAL STATISTICS" + (" (partial — stopped early)" if stopped_early else ""))
print("=" * 50)

# Report skipped samples
skipped = [r for r in sample_results if r.get('skipped')]
if skipped:
    print(f"\n⚠ Skipped {len(skipped)} sample(s) due to prompt length:")
    for i, r in enumerate(skipped):
        print(f"  - {r.get('skip_reason', 'unknown')}")

print(f"\nCompilation Statistics:")
print(f"  Total generation attempts: {total_attempts}")
if total_attempts > 0:
    print(f"  Successful compilations:   {total_successes} ({total_successes/total_attempts*100:.2f}%)")
    print(f"  Failed compilations:       {total_failures} ({total_failures/total_attempts*100:.2f}%)")

# compile@k (averaged across samples)
if sample_results:
    avg_compile_at_1 = statistics.mean(r['compile_at_1'] for r in sample_results)
    avg_compile_at_k = statistics.mean(r['compile_at_k'] for r in sample_results)
    print(f"\ncompile@k (unbiased, Chen et al. estimator, averaged over {len(sample_results)} samples):")
    print(f"  compile@1 = {avg_compile_at_1:.4f}")
    print(f"  compile@{K} = {avg_compile_at_k:.4f}")

# CodeBLEU on compiled samples
codebleu_maxes = [r['max_codebleu'] for r in sample_results if r['max_codebleu'] is not None]
if codebleu_maxes:
    min_cb = min(codebleu_maxes)
    max_cb = max(codebleu_maxes)
    avg_cb = statistics.mean(codebleu_maxes)
    std_cb = statistics.stdev(codebleu_maxes) if len(codebleu_maxes) > 1 else 0

    print(f"\nCodeBLEU Statistics (max per sample, compiled only):")
    print(f"  Samples with ≥1 compilation: {len(codebleu_maxes)}/{len(data_lines)}")
    print(f"  Min:     {min_cb:.4f}")
    print(f"  Max:     {max_cb:.4f}")
    print(f"  Average: {avg_cb:.4f}")
    print(f"  Std Dev: {std_cb:.4f}")

# Only write final CSVs if we completed all samples
if not stopped_early:
    if codebleu_maxes:
        # Write CodeBLEU statistics CSV
        codebleu_file = f"results/statistics/dart_statistics_{model_slug}_compiled.csv"
        with open(codebleu_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Min', 'Max', 'Average', 'Standard_Deviation',
                             'Samples_With_Success', 'Total_Samples'])
            writer.writerow([MODEL_NAME, min_cb, max_cb, avg_cb, std_cb,
                             len(codebleu_maxes), len(data_lines)])
        print(f"\nCodeBLEU statistics written to {codebleu_file}")

        # Write compilation + compile@k statistics CSV
        comp_file = f"dart_compilation_statistics_{model_slug}.csv"
        with open(comp_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Total_Attempts', 'Successful_Compilations',
                             'Failed_Compilations', 'Raw_Success_Rate',
                             'K_Value', 'Batch_Size',
                             'Avg_Compile@1', f'Avg_Compile@{K}',
                             'Skipped_Samples'])
            writer.writerow([MODEL_NAME, total_attempts, total_successes,
                             total_failures,
                             total_successes / total_attempts if total_attempts > 0 else 0,
                             K, BATCH_SIZE,
                             avg_compile_at_1, avg_compile_at_k,
                             len(skipped)])
        print(f"Compilation statistics written to {comp_file}")

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Checkpoint file {CHECKPOINT_FILE} removed (run complete).")
else:
    print(f"\nCheckpoint saved to {CHECKPOINT_FILE}. Re-run to resume.")
    if codebleu_maxes:
        print("(Final CSVs will be written once all samples complete.)")
    sys.exit(1)

if not codebleu_maxes:
    print("\nNo successful compilations across all samples!")

