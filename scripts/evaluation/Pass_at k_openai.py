from openai import OpenAI
import json
import subprocess
import tempfile
import os
import csv
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ──────────────────────────────────────────────────────────────
K = 10                # number of samples per problem
TEMPERATURE = 0.6     # higher temp for diversity across samples
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
BATCH_SIZE = 4        # concurrent API requests
DATA_FILE = "grpo_data"  # JSONL file
MODEL_NAME = "raafatabualazm/decompiler-v5"  # adjust to your deployed model name
DART_TIMEOUT = 30     # seconds per test execution
BASE_URL = "http://127.0.0.1:18000/v1"
API_KEY = "sk-45co41dpxtvdka"

# Checkpoint file for resume support
CHECKPOINT_FILE = f"results/cache/checkpoint_{MODEL_NAME.replace('/', '_')}.json"
# Force-skip only this exact JSONL entry and score it as 0 passed.
FORCED_ZERO_SOURCE_LINE = 152
FORCED_ZERO_FILENAME = "160.dart"
# ────────────────────────────────────────────────────────────────────────

client_kwargs = {"api_key": API_KEY}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL
client = OpenAI(**client_kwargs)

# ── System / Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a reverse engineering expert with advanced knowledge in assembly and Dart. "
    "Please convert the provided assembly code to idiomatic and clear Dart code. "
    "Before answering, think carefully about the task and create a step-by-step chain "
    "of thoughts to ensure a logical and accurate response. "
    "Return ONLY the Dart code inside a ```dart``` code block."
)

USER_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and Dart. 
Please convert the following assembly code to idiomatic and clear Dart code. 

### Assembly:
{}

### Response:
"""


# ── Checkpoint helpers ──────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """Load existing checkpoint if present. Returns {task_id: row_dict}."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        print(f"Loaded checkpoint with {len(ckpt)} completed tasks from {CHECKPOINT_FILE}")
        return ckpt
    return {}


def save_checkpoint(completed: dict):
    """Atomically write checkpoint to disk."""
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completed, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)


# ── Helpers ─────────────────────────────────────────────────────────────
def strip_main_and_imports(code: str) -> str:
    """Remove void main() { ... } block and import lines from generated code."""
    code = re.sub(r"^import\s+.*;\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)

    main_match = re.search(r"void\s+main\s*\([^)]*\)\s*\{", code)
    if main_match:
        start = main_match.start()
        depth = 0
        i = main_match.end() - 1
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    code = code[:start] + code[i + 1:]
                    break
            i += 1

    return code.strip()


def extract_dart_code(response: str) -> str:
    """Pull the Dart code block out of the model response."""
    try:
        parts = response.split("### Response:")
        text = parts[1] if len(parts) >= 2 else parts[0]
        if "```dart" in text:
            return text.split("```dart")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()
    except Exception:
        return response.strip()


def run_dart_test(generated_code: str, test_code: str, task_id: str) -> bool:
    """Combine generated function + test harness, run it, return True if all assertions pass."""
    clean_code = strip_main_and_imports(generated_code)
    imports = set(re.findall(r"^import\s+.*;\s*$", generated_code, re.MULTILINE))
    full_source = "\n".join(imports) + "\n\n" + clean_code + "\n\n" + test_code

    tmp_dir = tempfile.mkdtemp(prefix=f"dart_passk_{task_id}_")
    dart_file = os.path.join(tmp_dir, "test.dart")

    with open(dart_file, "w", encoding="utf-8") as f:
        f.write(full_source)

    try:
        result = subprocess.run(
            ["dart", "run", dart_file],
            capture_output=True,
            text=True,
            timeout=DART_TIMEOUT,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [task {task_id}] TIMEOUT")
        return False
    except Exception as e:
        print(f"  [task {task_id}] exec error: {e}")
        return False
    finally:
        try:
            os.remove(dart_file)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def call_openai(assembly: str) -> str:
    """Single API call → one completion. Raises on failure so caller can abort."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(assembly)},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def generate_samples(assembly: str, n_samples: int) -> list[str]:
    """Generate n_samples decompiled Dart outputs via parallel API calls.
    Raises immediately if any API call fails."""
    samples: list[str] = []

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        futures = [pool.submit(call_openai, assembly) for _ in range(n_samples)]
        for fut in as_completed(futures):
            samples.append(fut.result())  # re-raises exceptions from the thread

    return samples[:n_samples]


# ── Unbiased pass@k estimator (Chen et al., 2021) ──────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def should_force_zero(entry: dict) -> bool:
    """Return True only for the configured JSONL line/filename pair."""
    source_line = entry.get("_source_line")
    filename = entry.get("filename")
    return source_line == FORCED_ZERO_SOURCE_LINE and filename == FORCED_ZERO_FILENAME


def build_zero_row(task_id: str) -> dict:
    """Create a result row representing 0/K passed and pass@k = 0.0."""
    row = {"task_id": task_id, "n": K, "c": 0}
    for k_val in [1, 5, 10]:
        if k_val <= K:
            row[f"pass@{k_val}"] = 0.0
    return row


# ── Load data (Dart only) ──────────────────────────────────────────────
data = []
for fname in [DATA_FILE, DATA_FILE + ".jsonl", DATA_FILE + ".json"]:
    if os.path.exists(fname):
        with open(fname, encoding="utf-8") as f:
            for source_line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                lang = entry.get("lang", entry.get("language", "")).lower()
                if lang == "dart":
                    entry["_source_line"] = source_line_no
                    data.append(entry)
        print(f"Loaded {len(data)} Dart entries from {fname}")
        break
else:
    raise FileNotFoundError(f"Cannot find {DATA_FILE}")

# ── Load checkpoint ────────────────────────────────────────────────────
completed = load_checkpoint()

# ── Main evaluation loop ───────────────────────────────────────────────
results = []

# Restore already-completed results in order
for idx, entry in enumerate(data):
    task_id = entry.get("task_id", str(idx))
    if should_force_zero(entry):
        row = build_zero_row(task_id)
        results.append(row)
        completed[task_id] = row
        continue
    if task_id in completed:
        results.append(completed[task_id])

skipped = len(results)
if skipped:
    print(f"Resuming: skipping {skipped} already-completed tasks\n")

for idx, entry in enumerate(data):
    task_id = entry.get("task_id", str(idx))

    # Force-skip specific sample(s) with 0-pass scores.
    if should_force_zero(entry):
        row = build_zero_row(task_id)
        completed[task_id] = row
        save_checkpoint(completed)
        print(f"\n[{idx+1}/{len(data)}] Task {task_id}  —  forced skip (0/{K} passed)")
        continue

    # Skip already-completed tasks
    if task_id in completed:
        continue

    assembly = entry["assembly"]
    tests = entry["tests"]

    print(f"\n[{idx+1}/{len(data)}] Task {task_id}  —  generating {K} samples …")

    try:
        samples = generate_samples(assembly, K)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"API FAILURE on task {task_id} (index {idx+1}/{len(data)}): {e}")
        print(f"Checkpoint saved with {len(completed)} completed tasks.")
        print(f"Re-run the script to resume from this task.")
        print(f"{'='*60}")
        save_checkpoint(completed)
        sys.exit(1)

    n_passed = 0
    for s_idx, raw_output in enumerate(samples):
        code = extract_dart_code(raw_output)
        ok = run_dart_test(code, tests, task_id)
        if ok:
            n_passed += 1
        status = "✓" if ok else "✗"
        print(f"  sample {s_idx+1}/{K}: {status}")

    row = {"task_id": task_id, "n": K, "c": n_passed}
    for k_val in [1, 5, 10]:
        if k_val <= K:
            row[f"pass@{k_val}"] = pass_at_k(K, n_passed, k_val)
    results.append(row)

    # Persist after every task
    completed[task_id] = row
    save_checkpoint(completed)

    print(f"  ⇒ {n_passed}/{K} passed  |  " +
          "  ".join(f"pass@{k}: {row.get(f'pass@{k}', 'N/A'):.4f}"
                    for k in [1, 5, 10] if f"pass@{k}" in row))

# ── Aggregate & save ───────────────────────────────────────────────────
k_values = [k for k in [1, 5, 10] if k <= K]

print("\n" + "=" * 60)
print("AGGREGATE RESULTS")
print("=" * 60)
for k_val in k_values:
    scores = [r[f"pass@{k_val}"] for r in results if f"pass@{k_val}" in r]
    avg = np.mean(scores) if scores else 0.0
    print(f"  pass@{k_val} = {avg:.4f}  (over {len(scores)} tasks)")

csv_file = f"results/pass_at_k/dart_pass_at_k_{MODEL_NAME.replace('/', '_')}.csv"
fieldnames = ["task_id", "n", "c"] + [f"pass@{k}" for k in k_values]
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

summary_file = f"results/pass_at_k/dart_pass_at_k_summary_{MODEL_NAME.replace('/', '_')}.csv"
with open(summary_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value", "n_tasks"])
    for k_val in k_values:
        scores = [r[f"pass@{k_val}"] for r in results]
        writer.writerow([f"pass@{k_val}", f"{np.mean(scores):.6f}", len(scores)])

print(f"\nPer-task results  → {csv_file}")
print(f"Summary           → {summary_file}")

# Clean up checkpoint on successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"Checkpoint removed (all tasks complete).")

